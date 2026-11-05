// Service worker: runs the analysis so it survives the popup closing, and handles the hotkey.
import { analyze } from "./advisor.js";

chrome.commands.onCommand.addListener((cmd) => { if (cmd === "analyze") analyze(); });
chrome.runtime.onMessage.addListener((msg) => { if (msg?.type === "analyze") analyze(msg); });
