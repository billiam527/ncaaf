#!/usr/bin/env bash
# Check the CFBD key the pipeline would use: does it load, does it work, and is
# it still the one sitting in public git history?
#
# Never prints the key. Run after rotating.
cd /home/bill/ncaaf
source /home/bill/.ncaaf/bin/activate

python - <<'PY'
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, '/home/bill/ncaaf/etl/collect/collect_cfbd_games')
from scrape_cfbd_data import load_cfbd_key

key = load_cfbd_key()
if not key:
    print("  NO KEY FOUND in $CFBD_API_KEY or ~/.cfbd_api_key")
    raise SystemExit(1)

src = 'the CFBD_API_KEY environment variable' if os.environ.get('CFBD_API_KEY') \
      else '~/.cfbd_api_key'
print(f"  loaded from {src}: {len(key)} chars, ends {key[-4:]}")

print("\n=== does it authenticate? ===")
req = urllib.request.Request(
    'https://api.collegefootballdata.com/conferences',
    headers={'Authorization': f'Bearer {key}', 'accept': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print(f"  HTTP {r.status} - the key works")
except Exception as e:
    code = getattr(e, 'code', None)
    print(f"  FAILED: {e}")
    if code == 401:
        print("  401 means the key is not valid. If you just rotated, make sure")
        print("  the new value replaced the file contents rather than being")
        print("  appended, and that there is no trailing whitespace.")
    raise SystemExit(1)

print("\n=== is this key exposed in git history? ===")
out = subprocess.run(['git', 'log', '--all', '--format=%h %ad %s',
                      '--date=short', '-S', key],
                     capture_output=True, text=True, cwd='/home/bill/ncaaf')
hits = [l for l in out.stdout.strip().split('\n') if l]
if hits:
    print(f"  *** STILL EXPOSED in {len(hits)} commit(s) ***")
    for l in hits:
        print(f"      {l}")
    print("  This key must not be used. Rotate again and do not reuse a value")
    print("  that has ever been committed.")
    raise SystemExit(1)
print("  clean: this value appears in no commit, on any branch")

print("\n=== and it is not in the working tree ===")
out = subprocess.run(['grep', '-rIl', '--exclude-dir=.git', key, '.'],
                     capture_output=True, text=True, cwd='/home/bill/ncaaf')
if out.stdout.strip():
    print("  *** found in tracked files ***")
    print(out.stdout)
    raise SystemExit(1)
print("  clean")

print("\n  Key rotated correctly.")
PY
echo
echo "=== file permissions ==="
ls -l ~/.cfbd_api_key 2>/dev/null || echo "  ~/.cfbd_api_key not present"
