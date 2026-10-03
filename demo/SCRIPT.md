# Filing Lens: 75-second demo script (phone)

**Setup before recording:** open the Pages URL in Chrome on the phone. On Wi-Fi, tap **Download model** and wait for 100% (about 2 GB for Qwen2.5-3B on WebGPU; a phone without WebGPU gets Qwen2.5-1.5B on CPU instead). Add to Home Screen. Put `tests/fixtures/BESTBUY_2023_10K.pdf` in Downloads. Turn on airplane mode.

The Playwright recording (`demo/demo.webm`, headless Chromium at Pixel 7 size) follows the same beats, with the generator stubbed by recorded responses because the build machine can't download weights. Its upload step runs single-threaded and takes about 100 s, so cut that wait in the edit.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:08 | Home screen, model card shows "Qwen2.5-3B on-device (WebGPU)", pill "0 doc bytes sent" | "Retail investors get 100-page filings and chatbots that guess. Filing Lens runs a local 3B model, Qwen 2.5, on this phone, and the filing never leaves it." |
| 0:08–0:20 | **Open a filing PDF** → Best Buy FY2023 10-K → "Reading page N of 75" → "Embedding on-device…" | "I open Best Buy's 10-K. It's read page by page and indexed right here. I'm in airplane mode." |
| 0:20–0:35 | Ask *How much cash did operating activities provide in fiscal 2023?* → answer with **p.42** chip + confidence meter | "Every sentence has to cite a page the model was actually shown. The meter is a calibrated probability that the pages support an answer." |
| 0:35–0:45 | Tap **p.42** → the cash-flow statement renders | "One tap and I'm on the page: $1,824 million." |
| 0:45–0:58 | Ask *What was Walmart's capital expenditure in fiscal 2023?* → **abstain card** | "Now the wrong company. This is Best Buy's filing, so it says it can't answer instead of inventing a Walmart number." |
| 0:58–1:10 | Tap the privacy pill → 0 B sent, hosts table | "Zero request-body bytes left this page. The only download was the model, once." |
| 1:10–1:15 | Back to the answer | "Filing Lens: cited answers, or an honest 'I can't tell'. Your filing stays yours." |
