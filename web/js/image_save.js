import { app } from "../../scripts/app.js";
import { applyTextReplacements } from "../../scripts/utils.js";

// Fast Image Save (nodes/save_image.py) keeps the zlib-compressed metadata in APP15
// segments. The frontend reads the workflow only from PNG and WebP files, so
// dropping one of these JPEGs is handled here.
const NODE_TYPE = "FastImageSave";
const MARKER_SOS = 0xda;
const MARKER_APP15 = 0xef;
const SIGNATURE = new TextEncoder().encode("ComfyUI\0");

export function metadataChunks(buffer) {
    const view = new DataView(buffer);
    const chunks = [];
    let offset = 2;
    while (offset + 4 <= view.byteLength && view.getUint8(offset) === 0xff) {
        const marker = view.getUint8(offset + 1);
        if (marker === MARKER_SOS) break;
        const length = view.getUint16(offset + 2);
        const start = offset + 4;
        if (
            marker === MARKER_APP15 &&
            SIGNATURE.every((byte, i) => view.getUint8(start + i) === byte)
        ) {
            chunks.push(new Uint8Array(buffer, start + SIGNATURE.length, length - 2 - SIGNATURE.length));
        }
        offset += 2 + length;
    }
    return chunks;
}

export async function readMetadata(buffer) {
    const chunks = metadataChunks(buffer);
    if (!chunks.length) return null;
    const stream = new Blob(chunks).stream().pipeThrough(new DecompressionStream("deflate"));
    return JSON.parse(await new Response(stream).text());
}

app.registerExtension({
    name: "AlcheminePack.FastImageSave",
    // The frontend expands %date:...% only for its own save nodes.
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const widget = this.widgets.find((w) => w.name === "filename_prefix");
            widget.serializeValue = () => applyTextReplacements(app, widget.value);
            return result;
        };
    },
    init() {
        const handleFile = app.handleFile;
        app.handleFile = async function (file) {
            if (file?.type === "image/jpeg") {
                const metadata = await readMetadata(await file.arrayBuffer());
                const name = file.name.replace(/\.[^.]*$/, "");
                if (metadata?.workflow) {
                    return this.loadGraphData(JSON.parse(metadata.workflow), true, true, name);
                }
                if (metadata?.prompt) {
                    return this.loadApiJson(JSON.parse(metadata.prompt), name);
                }
            }
            return handleFile.apply(this, arguments);
        };
    },
});
