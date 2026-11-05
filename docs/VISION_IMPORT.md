# Screenshot import (vision)

"Import screenshot" in the Analysis board turns a picture of a Catan board into a PositionSpec.
It loads the position into the editor, outlines the fields it is unsure of, and shows the champion's quick suggestion for the next move.

## Use it for review, not for live play

This tool is for post-game review and for private games where every player knows about it and agreed.
Do not use it to get move advice during a live game against people who don't know about it.

It reads only the image file you upload. It does not connect to colonist.io or any other game site. It does not click, automate a browser or scrape anything.
With the colour backend the server makes no network requests. With the Gemini backend (below) the server sends the image to Google's Gemini API and nowhere else.
colonist.io's terms forbid scraping and automated play, and this tool does neither.
Keep real third-party screenshots, and the profiles you calibrate from them, in `data/screenshots/`. That folder is gitignored. Their artwork is not ours to commit.

## Using it

- **UI:** Analysis board, then **Import screenshot**. Pick a PNG, JPEG or WebP file of up to 8 MB.
  - The preview draws what was detected over your image: the hex outlines, terrain and number labels, the robber, harbors, pieces and roads. Low-confidence fields have red dashes.
  - **Load into editor** replaces the current position, and you can undo it.
  - The fields that were not read stay outlined in yellow dashes until you click **Dismiss**.
  - The colour backend does not read hands, card counts, dev cards, VP or the turn. Set them in the editor. The Gemini backend reads them when they are visible.
- **API:** `POST /api/analysis/import-image`, as `multipart/form-data`:
  - `image`: the file. Its part type must be `image/png`, `image/jpeg` or `image/webp`, and it may be at most 8 MB. Anything else returns 415 or 413.
  - `profile` (optional): a profile name. When it is absent, every profile is tried and the best board fit wins.
  - `num_players` (optional): `2`, `3` or `4`. The default is `4`.
  - `my_color` (optional): your colour name, such as `blue`, matched against `colors`. In the main game it makes that seat the viewer: you get the exact hand, the opponents get hidden counts, and when the turn was not read you are also the player to move. In setup it only adds a note when the next placement is not yours. A colour that isn't in `colors` (the colour backend never names colours) leaves the reading unchanged and says so in `override_note`.
  - `my_hand` (optional): your resource counts as `wood,brick,sheep,wheat,ore`, each 0 to 19, for example `1,0,2,0,0`. It is applied to the viewer in the main game and removes `hands` from `low_confidence.fields`. Bad values return 400. With either override the response has `override_note` (a string or `null`), and `errors` is recomputed.
  - `backend` (optional): `auto`, `llm` or `color`. The default `auto` uses Gemini when a key is configured. Without a key, `auto` and `llm` both fall back to the colour backend, and `backend_note` says so.
  - The response is `{position, errors, confidence, low_confidence, lattice, profile, backend, backend_note?, suggestion, suggestion_note?}`.
    The Gemini backend adds `colors` (colour to seat), `panel` (what it read off the side panels), `checks` (problems left after the repair call), `repaired` and `usage` (`{calls, input_tokens, output_tokens, total_tokens, usd}`). Its `lattice` can be `null`, and then the preview has no overlay.
  - A failed Gemini call (bad key, timeout, blocked answer) returns 502 with Google's error message. The key never appears in it.
  - `errors` comes from `validate_position_detailed`.
  - `suggestion` is `{policy, depth: "quick", candidates: [{action, label, policy_prob, value_estimate, search_score, rank}]}`. It is `null` when the position has errors or analysis is busy.
  - Limits checked at the boundary:
    - The request body is capped at 8 MB plus 64 KB. The global request guard raises its 1 MB cap for this path only.
    - The image itself is capped at 8 MB.
    - The format must be PNG, JPEG or WebP, checked by decoding the file, not by trusting its name.
    - The image may have at most 40 megapixels. This is checked before decoding.
    - Only one import runs at a time. Another request meanwhile gets 429.
- **Chrome extension:** `tools/chrome-advisor/` screenshots the visible tab when you ask and posts it here. See [its README](../tools/chrome-advisor/README.md). The server accepts that extension's origin on this endpoint only (`docs/SECURITY.md`).
- **Library:** `catan.vision.import_image(bytes, profile=None)` returns the same result without the suggestion. `catan.vision.llm.import_image_llm(bytes)` is the Gemini backend.

## Gemini backend (multimodal LLM)

The colour backend only knows renderers it was calibrated on. On a real colonist.io screenshot with the shipped `ours` profile it read the terrain and numbers wrong and returned 10 errors for roads not connected to buildings.
The Gemini backend asks a multimodal model to read the picture instead. It needs no calibration.

### Setup

1. Get a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env` at the repo root and set `GEMINI_API_KEY=...`. `.env` is gitignored.
   Optionally set `GEMINI_MODEL` (the default is `gemini-3.5-flash-lite`). The process environment wins over `.env`.
3. Import a screenshot. The `.env` file is read on every import, so a new key works without a restart. Restart the server only if you set the key in the environment instead.

`.env` is parsed with python-dotenv when it is installed, otherwise by a small built-in parser (`KEY=value`, `#` comments, optional quotes and `export`).

### Model and API (checked 2026-11-04)

- **Model:** `gemini-3.5-flash-lite`. Google lists it as stable, "the fastest, most cost-effective 3.5 model", and recommends 3.5 Flash-Lite or 3.8 Flash for new projects over `gemini-2.5-flash-lite`, whose access is limited to existing users. `gemini-3.1-flash-lite` is slightly cheaper but a generation older. If the reading is poor, set `GEMINI_MODEL=gemini-3.8-flash`: Google's most capable Flash, at 2.5× the input price. Sources: [models](https://ai.google.dev/gemini-api/docs/models), [pricing](https://ai.google.dev/gemini-api/docs/pricing).
- **API:** the REST method `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`, called with the standard library (`urllib`), so there is no new dependency. The key goes in the `x-goog-api-key` header, never in the URL. Google now recommends its newer Interactions API for new projects but says `generateContent` "remains fully supported". The `google-genai` SDK wraps the same endpoints. Sources: [generateContent reference](https://ai.google.dev/api/generate-content), [Interactions API](https://ai.google.dev/gemini-api/docs/interactions).
- **Structured output:** `generationConfig.responseMimeType = "application/json"` plus `responseJsonSchema` (`SCHEMA` in `catan/vision/llm.py`). The schema uses only the documented keywords: `type` (with `null` in a type array for optional fields), `enum`, `minimum`/`maximum`, `minItems`/`maxItems`, `required`. Google silently drops some keywords, such as string lengths, so the answer is still checked in code. Source: [structured output](https://ai.google.dev/gemini-api/docs/structured-output).
- **Image input:** inline base64 (`inlineData`) in PNG, JPEG, WebP, HEIC or HEIF, with at most 20 MB for the whole request. Our 8 MB upload cap is about 10.7 MB in base64, so it fits. An image costs 258 tokens per 768×768 tile: a 1920×1080 screenshot is about 6 tiles, roughly 1,550 tokens. Source: [image understanding](https://ai.google.dev/gemini-api/docs/image-understanding).

### What the model is asked for

The model never sees engine vertex or edge ids. It reports what it can see, and `catan/vision/llm.py` maps that onto `catan.engine.TOPOLOGY`:

- **Hexes:** 19 `{terrain, number}` in reading order: rows of 3, 4, 5, 4, 3 hexes, top to bottom, left to right. That is exactly the engine's hex id order: hexes are sorted by axial `(r, q)` with y down. The prompt spells out the rows, the centre (hex 9), both rings and some neighbour pairs.
- **Robber:** a hex index.
- **Harbors:** `{hex, side, type}`: the land hex it faces, the side toward the ship, and `3:1` or a resource.
- **Buildings:** `{color, kind, hex, corner}`, with corner `N|NE|SE|S|SW|NW`.
- **Roads:** `{color, hex, side}`, with side `NE|E|SE|SW|W|NW`.
- **Panel**, when visible: the side-panel players top to bottom, each with colour, cards, dev cards, VP, knights and road length. Also the user's colour, hand and dev cards (bottom left), the dice and whose turn it is.
- **board_points:** the pixel centres of hex 0 and hex 18, used for the preview overlay.

**Orientation.** Our topology is pointy-top: the corners of `hex_vertices` sit at 60·i − 30 degrees, y down, so the order is NE, SE, S, SW, NW, N. colonist.io draws pointy-top hexes too: a corner at the top of each tile, rows of 3-4-5-4-3. A side is named by the corners it joins (E joins NE and SE, and so on). If the model reports flat-top hexes, the import fails with a clear error.

`tests/vision/test_llm.py` checks the mapping in both directions for every hex, corner and side. Each corner and side points in the compass direction of its name. `vertex_name` and `edge_name` invert `corner_vertex` and `side_edge`. Positions survive a round trip through the model's terms.

**Seats.** Seat order is the side panel's order, top to bottom, followed by any colour seen only on the board. The response returns the table as `colors`. When the board is past setup, `viewer` is the user's colour and `current` is the turn colour. The user's hand and dev cards go to the viewer, and the opponents' card counts, dev counts and knights come from the panel. VP and road length are returned in `panel` but not put in the spec, because the engine recomputes them.

### Checks and the repair call

After the first call the reading is turned into a PositionSpec and checked:

- with `validate_position_detailed`, which checks terrain and token counts, harbors, the distance rule, roads connected to buildings, setup counts and hands;
- on a valid stand-in layout as well, because the validator stops at layout errors and piece errors should still reach the repair call;
- for problems found while mapping: a harbor not on the coast, two colours on one corner or side;
- for which hexes break a terrain or token quota, and whether two red numbers (6 or 8) are neighbours. colonist's balanced boards avoid neighbouring red numbers, so this only flags the hexes. It is not an error.

If anything is found, one repair call goes out. It carries the image, the previous JSON and the problems, each named in the model's own terms (`hex 9 side E`). The repaired reading is kept unless it has more validator errors than the first one. There are never more than two calls per import.

**Confidence.** A field the model read gets 0.8. A field named by a remaining problem gets 0.3, which is under the 0.6 cut, so the editor shows the usual dashed outline. Panel fields that were not visible get 0 and are listed in `low_confidence.fields`.

### Cost per import

Each response's `usage` reports the real token count and its price at the paid-tier rates in `PRICES`.
`gemini-3.5-flash-lite` costs $0.30 per 1M input tokens and $2.50 per 1M output tokens, thinking included.
One call is about 3,000 input tokens (prompt, schema and a 1080p image) and 1,000 to 4,000 output tokens (the JSON plus any thinking). That is about $0.004 to $0.011 per call.
With the repair call, an import costs at most about $0.025. With `gemini-3.8-flash` it is about 2.5× that.
These are estimates; no live call has been measured yet.

### Limits

- One image call and at most one repair call per import, with 180 s per call (`TIMEOUT`, the socket timeout of each call; `GEMINI_TIMEOUT` overrides it).
- The upload limits above still apply: 8 MB, PNG/JPEG/WebP, 40 megapixels, one import at a time. The image is decoded locally first, so a bad file never leaves the machine.
- At most 4 colours. Base board only.
- The model can misread. Check the outlined fields, and the seats, before you trust the suggestion.

### Privacy

- With a key configured, every import sends the screenshot to Google. Nothing else is sent, and nothing goes anywhere else. The UI says so next to the upload button. Choose "Colour tables (offline)" to keep the image local.
- On the free tier Google may use what you send to improve its products. On the paid tier it does not ([pricing](https://ai.google.dev/gemini-api/docs/pricing)). Don't send screenshots that show other people's names unless they agreed.
- The key lives only in `.env` or the environment. It is sent only in the request header to `generativelanguage.googleapis.com`, and it is never logged or returned.

## Phase guess

Nothing in a screenshot says whose turn it is, so the importer guesses from the pieces:

- A board that could be mid-setup becomes the next setup placement for the seat whose turn it is in snake order. Mid-setup means no cities, at most 2 settlements per seat, and as many roads as settlements or one fewer. The champion's suggestion is then a real opening move.
- Any other board becomes `MAIN` for seat 0 with an empty hand. Enter your hand to get a useful suggestion.

## Method

Each step is a small, inspectable classifier learned from labelled screenshots. There is no deep model.

1. **Board lattice.** A colour table gives every pixel a probability of being land.
   - A coarse-to-fine search over centre and hex radius maximises a score with four parts: land at two points of each of the 19 hexes, not-land on the ring of sea hexes around them, land just inside each of the 30 coastal edges, and not-land just outside them.
   - The coastline terms fix the scale.
   - Then `refine_lattice` snaps the fit to the number tokens. It finds each token by template match over small shifts, then least-squares fits (cx, cy, R).
   - The hex grid geometry follows Red Blob Games' hex guide: pointy-top, axial coordinates.
2. **Terrain.** Each pixel in a ring between the token and the pieces votes for its most likely terrain, using per-terrain colour histograms ("backprojection", Swain and Ballard 1991). The base-set quota (4/3/4/4/3/1) is then enforced by a greedy assignment.
3. **Numbers.** Normalised cross-correlation of the grey token disc against one mean template per number, under the token-quota constraint. OCR was not needed: the 10 token faces are a closed set, so a template matcher is simpler and needs no Tesseract install.
4. **Robber.** The hex with the most robber-coloured pixels.
5. **Harbors.** At each of the 30 coastal edges, an RGB patch at the marker position is compared with templates per type and with a "no harbor" template.
   - The 9 strongest markers are kept, with no two on adjacent coast edges.
   - Then the type quota is applied: 4 × 3:1 and one of each 2:1.
6. **Pieces.** A colour table maps pixels to background or seat 0..3.
   - **Settlements and cities:** a 10×10 map of the dominant seat colour around each vertex is classified by a class-balanced 5-NN as empty, settlement or city. Its shape, not its area, separates a city from a settlement with roads passing it.
   - **Roads:** the seat colour's share of a thin band along the middle of each edge, against a learned cut.
7. **Spec.** `board_to_spec` builds the PositionSpec, and `validate_position_detailed` checks it.

**Confidence** is between 0 and 1 for every field, and anything under 0.6 is flagged:

- terrain: the vote margin
- numbers and harbor types: the softmax over template scores
- robber: the margin over the second-best hex
- vertices: the k-NN vote share
- roads: the distance from the cut
- `lattice`: the board-fit score
- panel fields: always 0, because they are not read

**Dependencies:** numpy, plus Pillow to decode images.

- OpenCV is not used. Everything above is a few array operations on a fixed 19-hex lattice, and the `cv2` wheel in one dev environment failed to import (a missing libavif symbol). That is a dependency risk the task doesn't need.
- pytesseract is not used either, for the reason given under Numbers.

Prior work that shaped this:

- [Vieja/Catan-Image-Recognition](https://www.github.com/Vieja/Catan-Image-Recognition): HSV colour ranges per tile and per player piece on photos.
- [CMU 18-549 "Catan-omous Dealer"](https://course.ece.cmu.edu/~ece549/spring16/team08/website): edge detection plus Tesseract.
- [TU Delft CV board-game detection](https://hackmd.io/@937dn7BYQSSGBulQVdbfFQ/r1Np9GoaJx): synthetic training boards, which is also our test strategy.
- [Red Blob Games, hexagonal grids](https://www.redblobgames.com/grids/hexagons-v2/more-pixel-to-hex.html): lattice maths.

## Accuracy on our own app

The shipped profile `catan/vision/profiles/ours.npz` was calibrated on 9 screenshots of our own Analysis board (bot-game positions at 1280×800, 1440×900 and 1600×1000). On 5 held-out boards it read terrain, numbers, harbors and the robber 100% correctly and pieces 98.7% (153/155). A copy scaled to 0.7× and re-saved as JPEG at quality 75 still read all terrain and at least 18 of 19 numbers. A run takes about 1 s per image on one core.

These numbers say that the method works when the profile matches the renderer. They say nothing about colonist.io until it is calibrated, as described below.

## A real colonist.io screenshot: the calibration step

The shipped profile only knows our app's colours and shapes. A colonist.io screenshot needs its own profile, built from one or two screenshots you label yourself:

1. Take a screenshot of a finished or private game, uncropped and unrotated. Keep it in `data/screenshots/`.
2. Import it with the `ours` profile. It will be poor. Correct every field in the Analysis editor, then export the position (Import / export, then Download).
   - The labelled seats decide the colours: seat i is whatever colour seat i's pieces have in your labels.
   - Label 2 to 4 seats, and include at least one settlement, one city and one road.
   - colonist.io lets players pick colours, so a profile covers only the colours it saw. Calibrate on a game with the colours you care about, or on several screenshots.
3. Save the JSON next to the image with the same name (`shot1.png` and `shot1.json`). Then run:

   ```bash
   uv run python -m catan.vision.calibrate --out data/screenshots/colonist.npz \
       --robber-offset 0 0 --port-offset 0.62 data/screenshots/shot1.png [more pairs]
   ```

   The board lattice is found from your terrain labels alone (`fisher_scorer`), so you don't need to click any points.
   - `--robber-offset` is where the robber sits relative to its hex centre, in hex radii. Ours is (-0.5, 0.01): left of the token.
   - `--port-offset` is how far outside the coast the harbor marker is drawn.
   - Measure both offsets once from your screenshot. They are the only geometry the colour tables can't learn.
4. Profiles in `data/screenshots/*.npz` are picked up automatically by the endpoint. "Auto" tries them all, or you can pass `profile=colonist`.

What calibration can't fix:

- A board drawn in perspective or rotated. Only translation and scale are fitted.
- Pieces that hide most of a token or harbor marker.
- Tokens whose digits differ only by an underline on a skin whose pips are hidden.
- Expansions and non-standard maps. The 19-hex base layout and the base-set quotas are assumed.

## Limitations

- The colour backend never reads hands, card counts, dev cards, knights, VP or the turn. They get defaults with confidence 0 and are listed in `low_confidence.fields`.
- The phase is a guess from the piece counts, as described above.
- The assignment under the quotas is greedy. It is marked `shortcut:` in `catan/vision/detect.py`.
- k-NN vertex confidence can be high on a wrong answer (the held-out city above). Check the cities.
- Seat numbers come from colours. On colonist.io the seat order is not the colour order, so set "Analyze as" to your own seat.
