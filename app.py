import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# メンションされたらsay
@app.event("app_mention")
def message_mention(event, say):
  user = event["user"]
  thread_ts = event["ts"]
  say(thread_ts=thread_ts, text=f"Hello <@{user}>!")

# アプリを起動します
if __name__ == "__main__":
  SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
