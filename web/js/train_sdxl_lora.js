import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Live loss chart under the Train SDXL LoRA node. The node streams one
// "alchemine.train_progress" event per sd-scripts step ({node, step, total,
// loss}); the chart keeps the points for the node and redraws on each.
const NODE_TYPE = "TrainSdxlLora";
const EVENT = "alchemine.train_progress";
const HEIGHT = 180;

export function drawLossChart(canvas, points, total) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const pad = { l: 44, r: 10, t: 22, b: 22 };
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, w, h);
    ctx.font = "11px sans-serif";
    ctx.fillStyle = "#ccc";
    if (!points.length) {
        ctx.fillText("loss: waiting for the first step", pad.l, pad.t);
        return;
    }
    const last = points[points.length - 1];
    ctx.fillText(`step ${last.step}/${total}   avr_loss ${last.loss.toFixed(4)}`, pad.l, 14);
    const xs = points.map((p) => p.step), ys = points.map((p) => p.loss);
    const xMax = Math.max(total, xs[xs.length - 1]);
    const yMin = 0, yMax = Math.max(...ys) * 1.1 || 1;
    const X = (x) => pad.l + ((w - pad.l - pad.r) * x) / xMax;
    const Y = (y) => h - pad.b - ((h - pad.t - pad.b) * (y - yMin)) / (yMax - yMin);
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = yMin + ((yMax - yMin) * i) / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, Y(y)); ctx.lineTo(w - pad.r, Y(y)); ctx.stroke();
        ctx.fillStyle = "#888"; ctx.textAlign = "right";
        ctx.fillText(y.toFixed(3), pad.l - 4, Y(y) + 4);
    }
    ctx.textAlign = "center";
    for (const x of [0, xMax / 2, xMax]) ctx.fillText(String(Math.round(x)), X(x), h - 6);
    ctx.textAlign = "left";
    ctx.strokeStyle = "#4da3ff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(X(p.step), Y(p.loss)) : ctx.moveTo(X(p.step), Y(p.loss))));
    ctx.stroke();
}

// Two settings that INPUT_TYPES can only express as two widgets each are drawn
// as one row, the way comfyui-generator-pack draws its category rows:
//   noise_offset:  [toggle] noise_offset                   0.0357
//   train_length:  [epochs][steps] training length             22
//   save_steps:    [epochs][steps] save interval             200 (sample, log alike)
// The real widgets stay in node.widgets (the first one hidden, the value one
// replaced by the row under the same name), so saving and the prompt the
// backend receives are untouched. The value is typed in by clicking it.
function isLowQuality() {
    return (app.canvas?.ds?.scale || 1) <= 0.5;
}

function drawToggle(ctx, x, y, height, on) {
    const width = height * 1.5;
    const radius = height * 0.36;
    if (!isLowQuality()) {
        const alpha = ctx.globalAlpha;
        ctx.beginPath();
        ctx.roundRect(x + 4, y + 4, width - 8, height - 8, [height * 0.5]);
        ctx.globalAlpha = alpha * 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
        ctx.globalAlpha = alpha;
    }
    ctx.beginPath();
    const knobX = on ? x + width - radius - 5 : x + radius + 5;
    ctx.arc(knobX, y + height * 0.5, radius, 0, Math.PI * 2);
    ctx.fillStyle = on ? "#89B" : "#888";
    ctx.fill();
    return [[x, width]];
}

// One pill per choice, the chosen one lit. Returns the pills' [x, width].
function drawChoices(ctx, x, y, height, choices, chosen) {
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const bounds = [];
    for (const choice of choices) {
        const width = ctx.measureText(choice).width + 16;
        ctx.beginPath();
        ctx.roundRect(x + 2, y + 3, width, height - 6, [height * 0.5]);
        ctx.fillStyle = choice === chosen ? "#89B" : "rgba(255,255,255,0.12)";
        ctx.fill();
        ctx.fillStyle = choice === chosen ? "#111" : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
        ctx.fillText(choice, x + 2 + width / 2, y + height * 0.5);
        bounds.push([x, width + 4]);
        x += width + 4;
    }
    return bounds;
}

// spec: { drawLeft(ctx, x, y, h) -> [[x, width], ...], onLeft(index), label(value), inert(), name? }
function makeRow(node, valueWidget, spec) {
    const row = {
        type: "custom",
        name: valueWidget.name,
        options: valueWidget.options,
        get value() { return valueWidget.value; },
        set value(v) { valueWidget.value = v; },
        serializeValue() { return valueWidget.value; },
        computeSize(width) { return [width, LiteGraph.NODE_WIDGET_HEIGHT]; },
    };
    const min = valueWidget.options?.min ?? -Infinity;
    const max = valueWidget.options?.max ?? Infinity;
    const changed = () => {
        node.trainStart = null;
        node.updatePaths?.();
        node.setDirtyCanvas(true, true);
    };

    row.draw = function (ctx, node, width, y, height) {
        const margin = 15;
        const midY = y + height * 0.5;
        ctx.save();
        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.beginPath();
        ctx.roundRect(margin, y, node.size[0] - margin * 2, height, isLowQuality() ? [0] : [height * 0.5]);
        ctx.fill();
        if (!isLowQuality()) ctx.stroke();

        this.leftBounds = spec.drawLeft(ctx, margin + 6, y, height);
        const last = this.leftBounds.at(-1);
        const x = last[0] + last[1] + 5;
        if (isLowQuality()) { ctx.restore(); return; }
        if (spec.inert()) ctx.globalAlpha = app.canvas.editor_alpha * 0.4;

        ctx.font = "12px sans-serif";
        ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(spec.name ?? valueWidget.name, x, midY);
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "right";
        ctx.fillText(spec.label(valueWidget.value), node.size[0] - margin - 12, midY);
        ctx.restore();
    };

    row.mouse = function (event, pos, node) {
        if (event.type !== "pointerdown") return false;
        const hit = this.leftBounds?.findIndex(([x, w]) => pos[0] >= x && pos[0] <= x + w);
        if (hit >= 0) {
            spec.onLeft(hit);
            changed();
            return true;
        }
        if (spec.inert()) return true;
        app.canvas.prompt("Value", valueWidget.value, (v) => {
            const parsed = Number(v);
            if (!Number.isNaN(parsed)) {
                valueWidget.value = Math.min(max, Math.max(min, parsed));
                changed();
            }
        }, event);
        return true;
    };
    return row;
}

// Hides `leftWidget` and replaces `valueName` by a row that draws both.
function mergeRow(node, leftName, valueName, spec) {
    const leftWidget = node.widgets.find((w) => w.name === leftName);
    const valueIndex = node.widgets.findIndex((w) => w.name === valueName);
    leftWidget.computeSize = () => [0, -4];
    leftWidget.draw = () => {};
    leftWidget.type = "alchemine-hidden";
    node.widgets[valueIndex] = makeRow(node, node.widgets[valueIndex], spec(leftWidget));
}

function addRows(node) {
    mergeRow(node, "noise_offset", "noise_offset_value", (toggle) => ({
        drawLeft: (ctx, x, y, h) => drawToggle(ctx, x, y, h, toggle.value),
        onLeft: () => { toggle.value = !toggle.value; },
        label: (v) => Number(v).toFixed(4),
        inert: () => !toggle.value,
    }));
    mergeRow(node, "train_unit", "train_length", (unit) => ({
        drawLeft: (ctx, x, y, h) => drawChoices(ctx, x, y, h, unit.options.values, unit.value),
        onLeft: (index) => { unit.value = unit.options.values[index]; },
        label: (v) => String(Math.round(v)),
        inert: () => false,
        name: "training length",
    }));
    for (const [key, label] of [["save", "save"], ["sample", "sample"], ["logging", "log"]]) {
        mergeRow(node, `${key}_unit`, `${key}_steps`, (unit) => ({
            drawLeft: (ctx, x, y, h) => drawChoices(ctx, x, y, h, unit.options.values, unit.value),
            onLeft: (index) => { unit.value = unit.options.values[index]; },
            label: (v) => String(Math.round(v)),
            inert: () => false,
            name: `${label} interval`,
        }));
    }
    node.setSize(node.computeSize());
}

// Read-only lines above the loss chart showing where the node reads and writes:
// output/train/<name>/<version>, its mask/ folder (masked loss when present) and
// the checkpoints folder and step count, which are only known once a training starts.
const START_EVENT = "alchemine.train_start";

function addPathsWidget(node) {
    const box = document.createElement("div");
    box.style.cssText = "font: 11px/16px monospace; color: #bbb; padding: 4px 8px; white-space: pre; overflow: hidden; text-overflow: ellipsis;";
    // The frontend insets the element by `margin` on every side of the widget row
    // (10px by default), so the row must be taller than the four lines plus that.
    const widget = node.addDOMWidget("paths", "div", box, { serialize: false, margin: 4 });
    widget.computeSize = (width) => [width, 76];
    const byName = (name) => node.widgets.find((w) => w.name === name);
    node.trainStart = null;
    const update = () => {
        const name = byName("name")?.value || "<name>";
        const version = byName("version")?.value || "<version>";
        const base = `output/train/${name}/${version}`;
        const pending = "(set when a training starts)";
        const start = node.trainStart;
        const modelDir = start?.model_dir ?? `${base}/checkpoints ${pending}`;
        const byEpochs = byName("train_unit")?.value === "epochs";
        const detail = byEpochs ? ` (${start?.images} images × ${byName("train_length")?.value} epochs)` : "";
        const steps = start ? `${start.steps}${detail}` : pending;
        box.textContent = `output_dir: ${base}\nmask_dir:   ${base}/mask\nmodel_dir:  ${modelDir}\nsteps:      ${steps}`;
    };
    for (const w of [byName("name"), byName("version"), byName("train_unit"), byName("train_length")]) {
        const callback = w.callback;
        w.callback = function () { callback?.apply(this, arguments); node.trainStart = null; update(); };
    }
    const onConfigure = node.onConfigure;
    node.onConfigure = function () { onConfigure?.apply(this, arguments); update(); };
    node.updatePaths = update;
    update();
}

app.registerExtension({
    name: "AlcheminePack.TrainSdxlLora",
    setup() {
        api.addEventListener(START_EVENT, ({ detail }) => {
            const node = app.graph.getNodeById(detail.node);
            if (!node?.updatePaths) return;
            node.trainStart = detail;
            node.updatePaths();
        });
        api.addEventListener(EVENT, ({ detail }) => {
            const chart = app.graph.getNodeById(detail.node)?.lossChart;
            if (!chart) return;
            // sd-scripts prints each step twice (once to advance, once with the updated avr_loss):
            // the same step replaces the last point; a smaller step means a new run started.
            const last = chart.points.at(-1)?.step ?? -1;
            if (detail.step < last) chart.points.length = 0;
            else if (detail.step === last) chart.points.pop();
            chart.points.push({ step: detail.step, loss: detail.loss });
            chart.total = detail.total;
            drawLossChart(chart.canvas, chart.points, chart.total);
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            // A training seed is meant to stay put; the frontend defaults the control to randomize.
            this.widgets.find((w) => w.name === "control_after_generate").value = "fixed";
            addPathsWidget(this);
            addRows(this);
            const canvas = document.createElement("canvas");
            canvas.width = 480;
            canvas.height = HEIGHT;
            canvas.style.width = "100%";
            canvas.style.height = `${HEIGHT}px`;
            const widget = this.addDOMWidget("loss_chart", "canvas", canvas, { serialize: false });
            widget.computeSize = (width) => [width, HEIGHT + 8];
            this.lossChart = { canvas, points: [], total: 0 };
            drawLossChart(canvas, this.lossChart.points, this.lossChart.total);
        };
    },
});
