import requests
import websocket
import threading
import time
import json
from utils import CallState

class ARIController:
    def __init__(self, bot):
        self.bot = bot
        self.base_url = "http://127.0.0.1:8088/ari"
        self.auth = ("brain", "1234")
        self.app = "ai-bot"
        self.bridge_id = "auto_bridge"
        self.session = requests.Session()
        self.session.auth = self.auth
        self.lock = threading.Lock()
        self.processing_channels = set()

    def start(self):
        self._perform_cleanup()

        def run_ws():
            while not self.bot.should_terminate:
                print("[ARI] Connecting to ARI WebSocket...")
                ws_url = f"ws://127.0.0.1:8088/ari/events?app={self.app}&api_key=brain:1234"

                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=lambda ws: print("[ARI] Connected to ARI."),
                    on_error=lambda ws, err: print(f"[ARI] WebSocket Error: {err}"),
                    on_close=lambda ws, a, b: print("[ARI] ARI WebSocket Closed. Retrying in 2s..."),
                    on_message=self.on_message
                )

                self.ws.run_forever()
                if not self.bot.should_terminate:
                    time.sleep(2)

        thread = threading.Thread(
            target=run_ws,
            daemon=True
        )
        thread.start()

    def make_outbound_call(self, endpoint="6002"):
        try:
            print(f"[ARI] Dialing {endpoint} directly into Stasis...")

            resp = self.session.post(
                f"{self.base_url}/channels",
                json={
                    "endpoint": f"PJSIP/{endpoint}",
                    "app": "ai-bot"
                }
            )

            resp.raise_for_status()
            print("[ARI] Outbound call initiated.")

        except Exception as e:
            print(f"[ARI] Outbound call failed: {e}")

    def _perform_cleanup(self):
        print("[ARI] Performing startup cleanup of orphaned channels/bridges...")
        try:
            channels = self.session.get(f"{self.base_url}/channels").json()
            for ch in channels:
                if ch["name"].startswith("UnicastRTP"):
                    print(f"[ARI] Cleanup: Killing orphan channel {ch['id']}")
                    self.session.delete(f"{self.base_url}/channels/{ch['id']}")
                    time.sleep(0.1)

            try:
                self.session.delete(f"{self.base_url}/bridges/auto_bridge")
            except:
                pass
        except Exception as e:
            print(f"[ARI] Startup cleanup error: {e}")


    def _generate_and_save_mom(self):
        """
        Generates and saves the Minutes of Meeting in a background thread.
        """
        try:
            print("[ARI] Generating Minutes of Meeting (MOM)...")
            mom_text = self.bot.brain.generate_mom()
            if mom_text:
                with open("minutes_of_meeting.txt", "w") as f:
                    f.write(mom_text)
                print(f"[ARI] ✅ Minutes of Meeting saved to 'minutes_of_meeting.txt'")
            else:
                print("[ARI] ⚠️ MOM generation returned empty text.")
        except Exception as e:
            print(f"[ARI] ❌ Failed to generate/save MOM: {e}")

    def on_message(self, ws, message):
        event = json.loads(message)

        if event.get("type") == "StasisStart":
            channel = event["channel"]
            if channel["name"].startswith("UnicastRTP") or "ai-bot" in channel["name"]:
                return
            
            c_vars = channel.get("channelvars", {})
            stasis_args = event.get("args", [])
            
            is_outbound = (
                (c_vars.get("CALL_TYPE", "").lower() == "outbound") or 
                ("outbound" in stasis_args) or
                ("6002" in channel.get("name", ""))
            )
            call_type = "outbound" if is_outbound else "inbound"
            
            self.bot.call_type = call_type
            self.bot.channel_name = channel.get("name", "")
            print(f"[INFO] Call type detected: {call_type} (Channel: {self.bot.channel_name})")

            channel_id = channel["id"]
            channel_name = channel.get("name", "")

            with self.lock:
                if channel_id in self.processing_channels:
                    return
                self.processing_channels.add(channel_id)

            print(f"[ARI] StasisStart received for {channel_id} ({channel_name})")

            threading.Thread(
                target=self._handle_stasis_start,
                args=(channel_id, channel_name),
                daemon=True
            ).start()

        elif event.get("type") == "StasisEnd":
            channel = event.get("channel", {})
            name = channel.get("name", "")
            channel_id = channel.get("id")

            print(f"[ARI] StasisEnd received for {name}")

            if name.startswith("PJSIP"):
                print("[ARI] Phone channel ended. Generating MOM and performing cleanup.")

                threading.Thread(
                    target=self._generate_and_save_mom,
                    daemon=True
                ).start()

                try:
                    channels = self.session.get(f"{self.base_url}/channels").json()
                    for ch in channels:
                        if ch["name"].startswith("UnicastRTP"):
                            print(f"[ARI] Killing leftover ExternalMedia {ch['id']}")
                            self.session.delete(f"{self.base_url}/channels/{ch['id']}")
                            time.sleep(0.1)

                    bridges = self.session.get(f"{self.base_url}/bridges").json()
                    for br in bridges:
                        if br["id"].startswith("bridge_"):
                            print(f"[ARI] Deleting bridge {br['id']}")
                            self.session.delete(f"{self.base_url}/bridges/{br['id']}")
                            time.sleep(0.1)

                except Exception as e:
                    print(f"[ARI] Cleanup error: {e}")

                self.bot.remote_addr = None
                self.bot.asterisk_signaled_port = None
                self.bot.call_connected.clear()
                self.bot.set_state(CallState.LISTENING)


    def _cleanup_resources(self, bridge_id):

        with self.lock:
            try:
                print(f"[ARI] Cleaning up bridge {bridge_id}...")
                try:
                    resp = self.session.get(f"{self.base_url}/bridges/{bridge_id}")
                    if resp.status_code == 200:
                        bridge_details = resp.json()
                        channels = bridge_details.get("channels", [])
                        for ch_id in channels:
                            try:
                                ch_det = self.session.get(f"{self.base_url}/channels/{ch_id}").json()
                                if ch_det["name"].startswith("UnicastRTP"):
                                    print(f"[ARI] Destroying ExternalMedia channel {ch_id}")
                                    self.session.delete(f"{self.base_url}/channels/{ch_id}")
                            except:
                                pass

                        print(f"[ARI] Destroying bridge {bridge_id}")
                        self.session.delete(f"{self.base_url}/bridges/{bridge_id}")
                except Exception as e:
                    print(f"[ARI] Cleanup error: {e}")
            except Exception as e:
                print(f"[ARI] Critical Cleanup Error: {e}")

    def _safe_post(self, url, params=None, json_body=None, retries=5):
        for i in range(retries):
            try:
                resp = self.session.post(url, params=params, json=json_body, timeout=10)
                resp.raise_for_status()
                return resp
            except Exception as e:
                if i == retries - 1:
                    print(f"[ARI] Final Post Error for {url}: {e}")
                    raise e
                time.sleep(1.0)

    def _handle_stasis_start(self, channel_id, channel_name):
        with self.lock:
            try:
                if not channel_name.startswith("PJSIP"):
                    return

                print(f"[ARI] Answering channel {channel_id}...")
                self._safe_post(f"{self.base_url}/channels/{channel_id}/answer")

                time.sleep(1.0)

                unique_bridge_id = f"bridge_{channel_id}"
                print(f"[ARI] Creating isolation bridge {unique_bridge_id}...")

                self._safe_post(
                    f"{self.base_url}/bridges",
                    json_body={
                        "type": "mixing",
                        "bridgeId": unique_bridge_id
                    }
                )

                print(f"[ARI] Requesting ExternalMedia (slin) with JSON body...")
                ext_resp = self._safe_post(
                    f"{self.base_url}/channels/externalMedia",
                    json_body={
                        "app": self.app,
                        "external_host": "127.0.0.1:4000",
                        "format": "slin"
                    }
                )

                ext = ext_resp.json()
                external_id = ext["id"]
                print(f"[ARI] ExternalMedia ID: {external_id}")

                channelvars = ext.get("channelvars", {})
                signaled_port = channelvars.get("UNICASTRTP_LOCAL_PORT") or ext.get("local_port") or ext.get("port")
                if signaled_port:
                    self.bot.asterisk_signaled_port = int(signaled_port)

                print(f"[ARI] Linking channels to bridge...")
                self._safe_post(
                    f"{self.base_url}/bridges/{unique_bridge_id}/addChannel",
                    params={"channel": f"{channel_id},{external_id}"}
                )

                print("[ARI] Audio Bridge established.")
                self.bot.call_connected.set()
            except Exception as e:
                print(f"[ARI] Critical Error in StasisStart flow: {e}")
                self.bot.set_state(CallState.LISTENING)
