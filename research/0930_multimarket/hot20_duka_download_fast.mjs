import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const sourcePath = path.join(here, 'hot20_duka_download.mjs');
const generatedPath = path.join(here, 'hot20_duka_download_generated.mjs');
let source = fs.readFileSync(sourcePath, 'utf8');
source = source.replace(
  "const results = await runPool(TICKERS, 10);",
  "const selectedTickers = TICKERS.slice(0, 72);\nconst results = await runPool(selectedTickers, 18);"
);
source = source.replace(
  "requested: TICKERS.length,",
  "requested: selectedTickers.length,\n  source_universe_size: TICKERS.length,"
);
source = source.replace("concurrency: 10,", "concurrency: 18,");
source = source.replace(
  "console.log(JSON.stringify({ requested: manifest.requested, downloaded: manifest.downloaded, concurrency: manifest.concurrency }, null, 2));",
  "console.log(JSON.stringify({ requested: manifest.requested, source_universe_size: manifest.source_universe_size, downloaded: manifest.downloaded, concurrency: manifest.concurrency }, null, 2));"
);
fs.writeFileSync(generatedPath, source, 'utf8');
await import(pathToFileURL(generatedPath).href + `?v=${Date.now()}`);
