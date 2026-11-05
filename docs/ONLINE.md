# Playing online with friends

Online mode lets each friend play on their own device, from a private link. Any seat can be an AI agent, the Champion included. The bots run on your machine, as in a local game.

Code: `catan/server/online.py` (tokens, the request gate, room routes), `web/src/Room.tsx` (lobby). Tests: `tests/security/test_online_security.py`, `tests/server/test_online.py`.

## Start a room

1. Start the server in online mode on your computer:

   ```sh
   uv run python -m catan.server --online --port 8190
   ```

   It prints a line like `open http://127.0.0.1:8190/?admin=...`. Open that URL in your own browser, on this computer. The admin key is then kept in that browser, and the address bar drops it.
2. Make the server reachable for your friends (next section), and note the address it gets, such as `https://my-laptop.tail1234.ts.net`.
3. Go to **New game**, set up the seats (Human, or any AI agent), tick **Play online with friends**, then press **Create room**.
4. On the room page, paste your public address into **Address your friends use**. Then copy each human seat's link and send it to that friend only.
5. Each friend opens their link, sets a display name and presses **Take your seat**. To play a seat yourself, open its link too.

The room page shows who is connected. While another player is choosing, the game screen says "Waiting for <name>". Bots move while at least one player has the game screen open. To reconnect, open the same link again. Rooms are saved after every move to `results/saves/room-<id>.json`, so they survive a server restart.

## Make it reachable

Prefer Tailscale or a tunnel. **Don't forward a port on your router.** That exposes plain HTTP to the whole internet, with no TLS, and every scanner can find it. Tailscale and the tunnels below give your friends HTTPS without opening a port.

Keep the server on the default `--host 127.0.0.1` in all three cases. Tell the server the public hostname with `CATAN_ALLOWED_HOSTS`, because it refuses unknown `Host` headers (DNS-rebinding protection, see [SECURITY.md](SECURITY.md)).

### Tailscale (recommended)

Only devices on your tailnet can reach the game. Friends need a Tailscale account: share your machine with them from the admin console (**Machines**, then the machine's menu, then **Share**). See <https://tailscale.com/kb/1084/sharing>.

```sh
tailscale serve 8190
# prints https://<machine>.<tailnet>.ts.net, which proxies to http://127.0.0.1:8190
CATAN_ALLOWED_HOSTS=<machine>.<tailnet>.ts.net uv run python -m catan.server --online --port 8190
```

Stop sharing with Ctrl+C. `tailscale funnel 8190` publishes the same thing to the whole internet instead (stop it with `tailscale funnel 8190 off`). Then the seat links are the only protection, the same as with a tunnel.

### A tunnel (cloudflared or ngrok)

Anyone with the URL reaches the server, and the seat links are the only protection.

```sh
cloudflared tunnel --url http://localhost:8190
# prints a random https://<words>.trycloudflare.com; it changes on every start
CATAN_ALLOWED_HOSTS=<words>.trycloudflare.com uv run python -m catan.server --online --port 8190
```

Cloudflare's quick tunnels are for testing: no uptime guarantee, and at most 200 requests in flight. The game polls rather than using server-sent events, which quick tunnels don't support.

With ngrok (needs a free account):

```sh
ngrok config add-authtoken $YOUR_TOKEN
ngrok http 8190
```

Then start the server with `CATAN_ALLOWED_HOSTS` set to the hostname ngrok prints.

These commands come from the vendors' docs, checked 2026-11-04: <https://tailscale.com/kb/1312/serve>, <https://tailscale.com/kb/1311/tailscale-funnel>, <https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/> and <https://ngrok.com/docs/share-localhost/quickstart>.

If every move fails with "cross-origin request refused", the proxy is rewriting the `Host` header to `localhost`. Turn that rewrite off in the proxy's settings.

## Security model

**Whoever holds a seat link can play that seat.** Send each link to its player only, never to a group chat. The host page (the URL you land on after **Create room**) lists every link, so don't share it.

- **Seat links.** Each human seat gets a random 192-bit token (`secrets.token_urlsafe(24)`), and the room gets one more for the host page. The token sits in the URL fragment (`#/room/<id>/<token>`), which browsers never send to servers or in `Referer`. The app sends it to the API in an `X-Room-Token` header.
- **What a seat link can do.** It can view the game as its own seat, act for that seat only, step the bots, set its name, and read the replay after the game ends. The server ignores any `viewer=` it asks for. The view comes from the engine's `player_view(seat)` plus the per-seat `visible_to` event filter, the same code that hides AI hands in a local game. Other seats' hands, dev cards and private events (steals, dev-card draws) never leave the server. The test `test_no_private_info_reaches_another_seat` plays a whole game through the seat links and checks every response.
- **The host link** sees the lobby and every seat link, and gets a spectator view with no hands. It can't act.
- **Everything else needs the admin key.** That covers room creation, lab, analysis, saves, replays, plain games and the leaderboard. In online mode those routes need the admin key in `X-Catan-Admin`, whatever the client's address. A tunnel delivers every remote request from 127.0.0.1, so "comes from loopback" can't be trusted. The key comes from `CATAN_ADMIN_KEY`, or else `results/online_admin_key` (mode 600, created on first start).
- **Public routes:** `/api/health`, `/api/topology`, `/api/rules` and the static UI. They carry no game data.
- **Limits.** Each token gets 60 requests per 5 seconds, then 429; normal play uses about 6 per second at most. Room request bodies are capped at 8 KB (413), on top of the server-wide 1 MB cap. Names are 1 to 24 printable characters. The Host allowlist, cross-origin write refusal and no-CORS rules from [SECURITY.md](SECURITY.md) still apply.
- **Without `--online`** nothing changes: no room routes, no tokens, the local single-user server.

## Limits

- No accounts and no revocation. To shut someone out, start a new room. Deleting `results/saves/room-<id>.json` and restarting the server removes a room.
- Bots move only while some player has the game screen open. Bot speed is fixed at Normal online; there is no pause.
- Replays and analysis of a finished room game need the admin key, so friends can't open them from the victory screen.
- A room with search-based bots uses your CPU on every bot move. Online mode has no CPU budget beyond the per-token rate limit.
- Rooms count toward the server's 200 live-session cap, like any game.
