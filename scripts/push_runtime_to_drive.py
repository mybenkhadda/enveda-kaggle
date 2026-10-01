"""Upload the frozen CASMI runtime package to Google Drive through the Drive API (OAuth Installed-App flow).

Uploads ONLY:
    <ProjectRoot>/bundle/                      -> My Drive/<drive-folder>/bundle/        (recursive, relative paths kept)
    <ProjectRoot>/data/test.parquet            -> My Drive/<drive-folder>/competition/test.parquet
    <ProjectRoot>/data/sample_submission.csv   -> My Drive/<drive-folder>/competition/sample_submission.csv
and creates My Drive/<drive-folder>/results/ (never written to).

Never uploaded: src/, tests/, scripts/, notebooks/, outputs/, bundle_archive/, train.parquet, MOL_DEV / HOST artifacts,
QCR caches, __pycache__/, *.pyc, .ipynb_checkpoints/.

Secrets
    The OAuth client JSON is a SECRET INPUT FILE: pass --credentials <path> or set GOOGLE_DRIVE_CLIENT_SECRET.
    It is only handed to google-auth-oauthlib; this script never reads, prints or copies its contents.
    The OAuth token is stored OUTSIDE the repository (default %USERPROFILE%/.enveda/google_drive_token.json).
    Both paths are refused if they sit inside the project root (which is also synced by OneDrive).

Scope
    https://www.googleapis.com/auth/drive.file -- the app can only see and change files/folders it created itself.
    Consequence: a folder named <drive-folder> that was created some other way (Drive for desktop, the web UI, the old
    export_runtime_to_drive.ps1) is INVISIBLE to this app, and a second folder with the same name will be created.

Re-runs
    Existing Drive files (same name + same parent) are updated in place by file ID -- never duplicated.
    Same size -> SKIP (files <= 64 MB are additionally MD5-compared, cheap); different size -> update; --force -> update.
    If Drive holds a DIFFERENT config.json (another bundle), every same-size file is MD5-compared before skipping.

Dry run (--dry-run): validate the local package, print the plan; no Google authentication, nothing created remotely.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import mimetypes
import os
import sys
from pathlib import Path

DEFAULT_PROJECT_ROOT = r"C:\Users\myben\OneDrive\Documents\Enveda"
DEFAULT_DRIVE_FOLDER = "EnvedaCASMI"
DEFAULT_TOKEN = Path.home() / ".enveda" / "google_drive_token.json"
CREDENTIALS_ENV = "GOOGLE_DRIVE_CLIENT_SECRET"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

EXPECTED = {
    "bundle_version": "v2-A7",
    "model_id": "V1_TL_1K_TESTSIM_STRICT",
    "aggregator_name": "MOST_CONFIDENT_SPECTRUM",
}
COMPETITION_FILES = ("test.parquet", "sample_submission.csv")
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc"}
# uploaded last, so an interrupted upload never looks like a complete bundle on Drive
IDENTITY_FILES = ("manifest.json", "config.json")
MD5_CHEAP_BYTES = 64 * 1024 * 1024
FOLDER_MIME = "application/vnd.google-apps.folder"


class Stop(Exception):
    """Fatal, user-facing error (printed without a traceback)."""


# --------------------------------------------------------------------------------------------------------------------
# local helpers
# --------------------------------------------------------------------------------------------------------------------
def step(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    value = float(n)
    for unit in ("KB", "MB", "GB"):
        value /= 1024
        if value < 1024 or unit == "GB":
            break
    return f"{value:.2f} {unit}"


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class LocalFile:
    def __init__(self, path: Path, remote_parts: tuple[str, ...]):
        self.path = path
        self.remote_parts = remote_parts  # e.g. ("bundle", "code", "casmi", "x.py")
        self.size = path.stat().st_size
        self._md5: str | None = None

    @property
    def remote_rel(self) -> str:
        return "/".join(self.remote_parts)

    @property
    def md5(self) -> str:
        if self._md5 is None:
            self._md5 = file_hash(self.path, "md5")
        return self._md5


def validate_local(project_root: Path) -> tuple[dict, dict, list[LocalFile], list[LocalFile]]:
    bundle = project_root / "bundle"
    data = project_root / "data"

    step("1. local inputs")
    required = [bundle / "config.json", bundle / "manifest.json"] + [data / f for f in COMPETITION_FILES]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise Stop("missing local input(s):\n  " + "\n  ".join(missing))
    for p in required:
        print(f"  OK  {p}")

    step("2. frozen identity (bundle/config.json + manifest.json)")
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    man = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for key in ("bundle_version", "model_id", "aggregator_name", "CONFIG_HASH", "calibration_temperature"):
        print(f"  {key:<24}{cfg.get(key)}")
    problems = [f"config.{k} = {cfg.get(k)!r}, expected {v!r}" for k, v in EXPECTED.items() if cfg.get(k) != v]
    if not cfg.get("CONFIG_HASH"):
        problems.append("config.json has no CONFIG_HASH")
    if cfg.get("calibration_temperature") is None:
        problems.append("config.json has no calibration_temperature")
    if man.get("CONFIG_HASH") != cfg.get("CONFIG_HASH"):
        problems.append(f"manifest CONFIG_HASH {man.get('CONFIG_HASH')!r} != config CONFIG_HASH {cfg.get('CONFIG_HASH')!r}")
    if problems:
        raise Stop("the local bundle is not the frozen runtime bundle:\n  " + "\n  ".join(problems))
    print("  frozen identity OK")

    step("3. manifest completeness (local; presence only)")
    man_files = man.get("files")
    if not isinstance(man_files, dict) or not man_files:
        raise Stop('manifest.json has no "files" map')
    missing = [rel for rel in man_files if not (bundle / rel).is_file()]
    print(f"  manifest files present: {len(man_files) - len(missing)}/{len(man_files)}")
    if missing:
        raise Stop("local bundle is incomplete; first missing files:\n  " + "\n  ".join(missing[:20]))

    bundle_files: list[LocalFile] = []
    for dirpath, dirnames, filenames in os.walk(bundle):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix in SKIP_SUFFIXES or p.is_symlink():
                continue
            rel = p.relative_to(bundle).parts
            bundle_files.append(LocalFile(p, ("bundle",) + rel))
    # identity files last (stable sort keeps the walk order for everything else)
    top_level_identity = {("bundle", n): i for i, n in enumerate(IDENTITY_FILES)}
    bundle_files.sort(key=lambda f: top_level_identity.get(f.remote_parts, -1))
    comp_files = [LocalFile(data / f, ("competition", f)) for f in COMPETITION_FILES]
    cfg["_manifest_files"] = sorted(man_files)
    return cfg, man, bundle_files, comp_files


def print_plan(drive_folder: str, bundle_files: list[LocalFile], comp_files: list[LocalFile]) -> None:
    step("4. upload plan")
    dirs: dict[str, list[int]] = {}
    for f in bundle_files:
        d = "/".join(f.remote_parts[:-1])
        for i in range(1, len(f.remote_parts) - 1):  # list folders that only hold sub-folders too
            dirs.setdefault("/".join(f.remote_parts[:i]), [0, 0])
        dirs.setdefault(d, [0, 0])
        dirs[d][0] += 1
        dirs[d][1] += f.size
    print(f"  My Drive/{drive_folder}/")
    for d in sorted(dirs):
        depth = d.count("/")
        print(f"  {'    ' * (depth + 1)}{d.split('/')[-1]}/   ({dirs[d][0]} files, {human(dirs[d][1])})")
    print(f"      competition/")
    for f in comp_files:
        print(f"          {f.remote_parts[-1]}   ({human(f.size)})")
    print(f"      results/   (created empty, never written)")
    total = bundle_files + comp_files
    print(f"\n  files total: {len(total)}   (bundle {len(bundle_files)}, competition {len(comp_files)})")
    print(f"  bytes total: {sum(f.size for f in total):,}   ({human(sum(f.size for f in total))})")


# --------------------------------------------------------------------------------------------------------------------
# Google Drive
# --------------------------------------------------------------------------------------------------------------------
def authenticate(credentials_path: Path, token_path: Path):
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if token_path.is_file():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except (ValueError, json.JSONDecodeError):
            print("  stored token is unreadable -- a new authorization is needed")
            creds = None
        if creds is not None and not creds.has_scopes(SCOPES):
            print("  stored token has different scopes -- a new authorization is needed")
            creds = None

    if creds is not None and creds.valid:
        print("  reusing stored token")
    elif creds is not None and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            print("  stored token refreshed")
        except RefreshError:
            print("  token refresh was rejected (revoked or expired) -- a new authorization is needed")
            creds = None
    else:
        creds = None

    if creds is None:
        print("  opening the browser for Google authorization ...")
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
        creds = flow.run_local_server(
            port=0,
            authorization_prompt_message="  If the browser does not open, visit the URL printed below.\n  {url}",
            success_message="Authorization complete. You can close this tab.",
        )

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    try:
        os.chmod(token_path, 0o600)
    except OSError:
        pass
    return creds


class Drive:
    def __init__(self, creds):
        from googleapiclient.discovery import build

        self.svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._children: dict[str, dict[str, dict]] = {}   # parent id -> {"f:<name>"|"d:<name>": meta}
        self._folders: dict[tuple[str, ...], str] = {}

    def account_email(self) -> str | None:
        try:
            return self.svc.about().get(fields="user(emailAddress)").execute()["user"]["emailAddress"]
        except Exception:
            return None

    def children(self, parent_id: str, refresh: bool = False) -> dict[str, dict]:
        if refresh or parent_id not in self._children:
            out: dict[str, dict] = {}
            dupes: set[str] = set()
            token = None
            while True:
                resp = self.svc.files().list(
                    q=f"'{parent_id}' in parents and trashed=false",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, size, md5Checksum)",
                    pageSize=1000,
                    pageToken=token,
                ).execute(num_retries=5)
                for f in resp.get("files", []):
                    key = ("d:" if f["mimeType"] == FOLDER_MIME else "f:") + f["name"]
                    if key in out:
                        dupes.add(key[2:])
                        continue
                    out[key] = f
                token = resp.get("nextPageToken")
                if not token:
                    break
            for name in sorted(dupes):
                print(f"  WARNING: duplicate Drive entries named {name!r} under one folder -- using the first one")
            self._children[parent_id] = out
        return self._children[parent_id]

    def find_folder(self, parent_id: str, name: str) -> str | None:
        meta = self.children(parent_id).get("d:" + name)
        return meta["id"] if meta else None

    def ensure_folder(self, path: tuple[str, ...], root_id: str = "root") -> str:
        """Resolve/create My Drive/<path...>, reusing existing folders (name + parent + folder mime + not trashed)."""
        parent = root_id
        for i in range(1, len(path) + 1):
            key = path[:i]
            if key in self._folders:
                parent = self._folders[key]
                continue
            fid = self.find_folder(parent, path[i - 1])
            if fid is None:
                meta = self.svc.files().create(
                    body={"name": path[i - 1], "mimeType": FOLDER_MIME, "parents": [parent]},
                    fields="id, name, mimeType",
                ).execute(num_retries=5)
                fid = meta["id"]
                self.children(parent)["d:" + path[i - 1]] = meta
                self._children[fid] = {}
                print(f"  created folder {'/'.join(key)}")
            self._folders[key] = fid
            parent = fid
        return parent

    def find_file(self, parent_id: str, name: str) -> dict | None:
        return self.children(parent_id).get("f:" + name)

    def upload(self, local: LocalFile, parent_id: str, existing: dict | None, chunk_bytes: int, label: str) -> dict:
        from googleapiclient.http import MediaFileUpload

        mime = mimetypes.guess_type(local.path.name)[0] or "application/octet-stream"
        fields = "id, name, size, md5Checksum"
        if local.size == 0:
            media = MediaFileUpload(str(local.path), mimetype=mime, resumable=False)
        else:
            media = MediaFileUpload(str(local.path), mimetype=mime, resumable=True, chunksize=chunk_bytes)
        if existing:
            request = self.svc.files().update(fileId=existing["id"], media_body=media, fields=fields)
        else:
            request = self.svc.files().create(body={"name": local.path.name, "parents": [parent_id]},
                                              media_body=media, fields=fields)
        if local.size == 0:
            result = request.execute(num_retries=5)
        else:
            result = None
            while result is None:
                status, result = request.next_chunk(num_retries=5)
                if status is not None:
                    print(f"\r  {label} -- {int(status.progress() * 100)}%   ", end="", flush=True)
        print(f"\r  {label} -- 100%   ", flush=True)
        self.children(parent_id)["f:" + local.path.name] = result
        return result


# --------------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upload the frozen CASMI runtime package to Google Drive (API, OAuth).")
    ap.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    ap.add_argument("--credentials", default=None,
                    help=f"OAuth Installed-App client JSON (secret; keep it outside the repo). "
                         f"Falls back to the {CREDENTIALS_ENV} environment variable.")
    ap.add_argument("--token", default=str(DEFAULT_TOKEN), help="OAuth token cache (outside the repo).")
    ap.add_argument("--drive-folder", default=DEFAULT_DRIVE_FOLDER, help="top-level folder under My Drive")
    ap.add_argument("--dry-run", action="store_true", help="validate + plan only; no authentication, no upload")
    ap.add_argument("--force", action="store_true", help="upload/update every file regardless of size / MD5")
    ap.add_argument("--chunk-size-mb", type=int, default=32, help="resumable upload chunk size (MB, >= 1)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    token_path = Path(os.path.expandvars(os.path.expanduser(args.token))).resolve()
    cred_arg = args.credentials or os.environ.get(CREDENTIALS_ENV)
    credentials_path = Path(os.path.expandvars(os.path.expanduser(cred_arg))).resolve() if cred_arg else None
    if args.chunk_size_mb < 1:
        raise Stop("--chunk-size-mb must be >= 1")
    chunk_bytes = args.chunk_size_mb * 1024 * 1024  # multiple of 256 KB, as the resumable protocol requires

    if args.dry_run:
        print("DRY RUN -- validation and planning only; no Google authentication, nothing uploaded.")
    print(f"project root: {project_root}")

    # secret-location checks (paths only -- the credential file is never opened here)
    step("0. secret locations")
    if credentials_path is None:
        if not args.dry_run:
            raise Stop(f"no OAuth client file: pass --credentials <path> or set {CREDENTIALS_ENV}")
        print("  credentials: not given (not needed for a dry run)")
    else:
        if is_inside(credentials_path, project_root):
            raise Stop("the OAuth client file is inside the project root (git + OneDrive). Move it outside, "
                       r"e.g. C:\Users\myben\secrets\, and pass that path.")
        if not credentials_path.is_file():
            raise Stop(f"OAuth client file not found: {credentials_path}")
        print(f"  credentials: {credentials_path}   (outside repo; contents not read)")
    if is_inside(token_path, project_root):
        raise Stop("--token points inside the project root; keep the token outside the repository")
    print(f"  token:       {token_path}   (outside repo)")

    cfg, man, bundle_files, comp_files = validate_local(project_root)
    manifest_rel = cfg.pop("_manifest_files")
    print_plan(args.drive_folder, bundle_files, comp_files)
    all_files = bundle_files + comp_files
    bytes_total = sum(f.size for f in all_files)

    if args.dry_run:
        print("\nDRY RUN COMPLETE — nothing uploaded.")
        return 0

    step("5. Google Drive authentication")
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        raise Stop("Google client libraries missing. Install: python -m pip install -r requirements-dev.txt")
    creds = authenticate(credentials_path, token_path)
    drive = Drive(creds)
    print("  Google Drive authentication successful")
    email = drive.account_email()
    if email:
        print(f"  account: {email}")

    step(f"6. Drive folders (My Drive/{args.drive_folder}/ ...)")
    root_id = drive.ensure_folder((args.drive_folder,))
    bundle_id = drive.ensure_folder((args.drive_folder, "bundle"))
    comp_id = drive.ensure_folder((args.drive_folder, "competition"))
    results_id = drive.ensure_folder((args.drive_folder, "results"))
    print(f"  {args.drive_folder}: {root_id}")

    remote_cfg = drive.find_file(bundle_id, "config.json")
    local_cfg = next(f for f in bundle_files if f.remote_parts == ("bundle", "config.json"))
    deep = False
    if remote_cfg is not None and remote_cfg.get("md5Checksum") and remote_cfg["md5Checksum"] != local_cfg.md5:
        deep = True
        print("  Drive holds a DIFFERENT bundle config.json -> every same-size file is MD5-compared before skipping")

    step("7. upload (resumable; existing files updated in place by ID)")
    uploaded = updated = skipped = 0
    n = len(all_files)
    for i, f in enumerate(all_files, 1):
        parent = drive.ensure_folder((args.drive_folder,) + f.remote_parts[:-1])
        existing = drive.find_file(parent, f.path.name)
        label = f"[{i}/{n}] {f.remote_rel}"
        if existing is not None and not args.force:
            same_size = existing.get("size") is not None and int(existing["size"]) == f.size
            if same_size:
                remote_md5 = existing.get("md5Checksum")
                check_md5 = remote_md5 and (deep or f.size <= MD5_CHEAP_BYTES)
                if not check_md5 or remote_md5 == f.md5:
                    skipped += 1
                    print(f"  {label} -- SKIP (same size{', same MD5' if check_md5 else ''})")
                    continue
        drive.upload(f, parent, existing, chunk_bytes, label)
        if existing is None:
            uploaded += 1
        else:
            updated += 1

    step("8. verification (listing Drive again; sizes; MD5 of small identity files)")
    drive._children.clear()

    def remote_meta(parts: tuple[str, ...]) -> dict | None:
        parent = root_id
        for d in parts[:-1]:
            parent = drive.find_folder(parent, d)
            if parent is None:
                return None
        return drive.find_file(parent, parts[-1])

    local_by_rel = {f.remote_rel: f for f in all_files}
    problems: list[str] = []
    ok_manifest = 0
    for rel in manifest_rel:
        key = "bundle/" + rel
        local_size = (project_root / "bundle" / rel).stat().st_size
        meta = remote_meta(tuple(key.split("/")))
        if meta is None:
            problems.append(f"missing on Drive: {key}")
        elif int(meta.get("size", -1)) != local_size:
            problems.append(f"size differs on Drive: {key} ({meta.get('size')} vs {local_size})")
        else:
            ok_manifest += 1
    ok_comp = 0
    for f in comp_files:
        meta = remote_meta(f.remote_parts)
        if meta is None:
            problems.append(f"missing on Drive: {f.remote_rel}")
        elif int(meta.get("size", -1)) != f.size:
            problems.append(f"size differs on Drive: {f.remote_rel}")
        else:
            ok_comp += 1

    identity = [("bundle", "config.json"), ("bundle", "manifest.json"), ("competition", "sample_submission.csv")]
    identity_rows = []
    for parts in identity:
        lf = local_by_rel["/".join(parts)]
        meta = remote_meta(parts) or {}
        md5_ok = meta.get("md5Checksum") == lf.md5 if meta.get("md5Checksum") else None
        if md5_ok is False:
            problems.append(f"MD5 differs on Drive: {lf.remote_rel}")
        identity_rows.append({"file": lf.remote_rel, "local_sha256": file_hash(lf.path, "sha256"),
                              "local_md5": lf.md5, "drive_md5": meta.get("md5Checksum"), "md5_match": md5_ok})
        print(f"  {lf.remote_rel:<36} sha256 {identity_rows[-1]['local_sha256'][:16]}...  "
              f"drive md5 {'match' if md5_ok else ('n/a' if md5_ok is None else 'MISMATCH')}")

    print(f"\n  Remote bundle files: {ok_manifest}/{len(manifest_rel)}")
    print(f"  Competition files: {ok_comp}/{len(comp_files)}")
    status = "PASSED" if not problems else "FAILED"
    for p in problems[:30]:
        print(f"  PROBLEM: {p}")

    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "bundle_version": cfg.get("bundle_version"),
        "CONFIG_HASH": cfg.get("CONFIG_HASH"),
        "model_id": cfg.get("model_id"),
        "aggregator": cfg.get("aggregator_name"),
        "calibration_temperature": cfg.get("calibration_temperature"),
        "manifest_bundle_version": man.get("bundle_version"),
        "drive_root_folder": args.drive_folder,
        "drive_root_folder_id": root_id,
        "bundle_folder_id": bundle_id,
        "competition_folder_id": comp_id,
        "results_folder_id": results_id,
        "oauth_scope": SCOPES[0],
        "force": args.force,
        "files_total": len(all_files),
        "files_uploaded": uploaded,
        "files_updated": updated,
        "files_skipped": skipped,
        "bytes_total": bytes_total,
        "remote_manifest_files_ok": f"{ok_manifest}/{len(manifest_rel)}",
        "competition_files_ok": f"{ok_comp}/{len(comp_files)}",
        "identity_files": identity_rows,
        "verification_problems": problems,
        "verification_status": status,
    }
    report_path = project_root / "outputs" / "drive_upload_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  report: {report_path}")
    print(f"  uploaded {uploaded} | updated {updated} | skipped {skipped} | total {len(all_files)} ({human(bytes_total)})")
    print(f"\nVERIFICATION {status}")
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as exc:
        print(f"\nUPLOAD STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted -- re-run to resume (completed files are skipped).", file=sys.stderr)
        sys.exit(130)
