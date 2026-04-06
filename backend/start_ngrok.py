from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokHTTPError
from pyngrok.conf import PyngrokConfig
import os
import time

public_url = None
local_ngrok_path = os.path.join(os.path.dirname(__file__), "ngrok", "ngrok.exe")
pyngrok_config = PyngrokConfig(ngrok_path=local_ngrok_path)

try:
    public_url = ngrok.connect(8000, "http", pyngrok_config=pyngrok_config, pooling_enabled=True)
except PyngrokNgrokHTTPError as e:
    print("\n" + "="*60)
    print("⚠ ngrok endpoint conflict detected")
    print("="*60)
    print("\nTried starting with pooling enabled, but ngrok still rejected the tunnel.")
    print("\nAction:")
    print("1) Open https://dashboard.ngrok.com/endpoints")
    print("2) Stop the existing endpoint shown in the error")
    print("3) Re-run this script")
    print("\nFull error:")
    print(str(e))
    print("="*60 + "\n")
    raise

print("\n" + "="*60)
print("✓ ngrok tunnel is ACTIVE")
print("="*60)
print(f"\nYour public webhook URL for Vapi.ai:")
print(f"\n  {public_url}/webhook")
print("\nUse this in Vapi.ai webhook configuration")
print("\nKeep this window open while using Vapi.ai")
print("="*60 + "\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nClosing ngrok tunnel...")
    ngrok.disconnect(public_url)
