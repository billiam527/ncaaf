#!/usr/bin/env bash
# Why isn't the key working? Reports everything diagnostic about it without
# revealing it: length, what kinds of character are in it, whether anything
# invisible got in, whether it is still the old one, and what the API says.
cd /home/bill/ncaaf
source /home/bill/.ncaaf/bin/activate

echo '=== the file ==='
F=~/.cfbd_api_key
if [ ! -f "$F" ]; then
    echo "  ~/.cfbd_api_key does not exist"
else
    echo "  size on disk : $(stat -c%s "$F") bytes"
    echo "  permissions  : $(stat -c%A "$F")"
    echo "  modified     : $(stat -c%y "$F" | cut -d. -f1)"
    echo "  line count   : $(wc -l < "$F")   (1 means it ends with a newline)"
fi
echo
echo '=== is $CFBD_API_KEY set? it wins over the file ==='
if [ -n "$CFBD_API_KEY" ]; then
    echo "  YES - ${#CFBD_API_KEY} chars. The file is being ignored."
    echo "  If you meant to use the file, run: unset CFBD_API_KEY"
else
    echo "  not set, so the file is what gets used"
fi

python - <<'PY'
import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, '/home/bill/ncaaf/etl/collect/collect_cfbd_games')
from scrape_cfbd_data import KEY_FILE, load_cfbd_key

print("\n=== the value, described but not shown ===")
try:
    raw = open(KEY_FILE, 'rb').read()
except FileNotFoundError:
    print("  no file")
    raise SystemExit(1)

print(f"  raw bytes        : {len(raw)}")
stripped = raw.decode('utf-8', 'replace').strip()
print(f"  after stripping  : {len(stripped)} chars")
if len(raw) != len(stripped):
    extra = len(raw) - len(stripped)
    print(f"  -> {extra} byte(s) of surrounding whitespace, which is stripped "
          f"automatically, so this alone is not the problem")

for bad, why in ((r'["\']', 'quote marks'), (r'\s', 'internal whitespace'),
                 (r'^Bearer', 'a literal "Bearer " prefix'),
                 (r'[^A-Za-z0-9+/=_-]', 'characters outside the usual key alphabet')):
    if re.search(bad, stripped):
        print(f"  *** contains {why} - almost certainly the problem")

print(f"  charset          : "
      f"{'letters' if re.search(r'[A-Za-z]', stripped) else ''}"
      f"{'+digits' if re.search(r'[0-9]', stripped) else ''}"
      f"{'+symbols' if re.search(r'[^A-Za-z0-9]', stripped) else ''}")
print(f"  starts/ends      : {stripped[:3]}...{stripped[-3:]}")

print("\n=== is it actually new? ===")
out = subprocess.run(['git', 'log', '--all', '--format=%h %ad', '--date=short',
                      '-S', stripped],
                     capture_output=True, text=True, cwd='/home/bill/ncaaf')
if out.stdout.strip():
    print("  *** this is the OLD key - it is in git history ***")
    for l in out.stdout.strip().split('\n'):
        print(f"      {l}")
    print("  The file was not actually replaced. Check you used > and not >>,")
    print("  and that the shell did not expand anything in the value.")
else:
    print("  good: this value is in no commit, so it is a genuinely new key")

print("\n=== what the API says ===")
key = load_cfbd_key()
req = urllib.request.Request(
    'https://api.collegefootballdata.com/conferences',
    headers={'Authorization': f'Bearer {key}', 'accept': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print(f"  HTTP {r.status} - the key works")
except Exception as e:
    code = getattr(e, 'code', None)
    body = ''
    try:
        body = e.read().decode()[:200]
    except Exception:
        pass
    print(f"  HTTP {code}: {e}")
    if body:
        print(f"  response: {body}")
    print({
        401: "  401 = not accepted. Either the key is wrong, or it has not been\n"
             "  activated yet - CFBD keys can take a few minutes after the email.",
        403: "  403 = accepted but not authorised for this endpoint, which would\n"
             "  point at a tier or entitlement issue rather than the value.",
        429: "  429 = rate limited. The key is fine; you are over quota.",
    }.get(code, "  Not an auth code - could be network or the API being down."))
PY
