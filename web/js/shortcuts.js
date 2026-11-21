import { app } from "../../scripts/app.js";

// The queue mode lives in the frontend's pinia store, which is not exported to extensions.
function queueSettings() {
    const pinia = document.getElementById("vue-app").__vue_app__.config.globalProperties.$pinia;
    return pinia._s.get("queueSettingsStore");
}

app.registerExtension({
    name: "AlcheminePack.Shortcuts",
    commands: [
        {
            id: "AlcheminePack.ToggleInstantRun",
            label: "Toggle Run (Instant)",
            function: async () => {
                const settings = queueSettings();
                if (settings.mode === "instant") {
                    settings.mode = "disabled";
                    return;
                }
                settings.mode = "instant";
                await app.extensionManager.command.execute("Comfy.QueuePrompt");
            },
        },
    ],
    keybindings: [
        { combo: { key: "c" }, commandId: "refresh-impact-wildcard" },
        { combo: { key: "i" }, commandId: "AlcheminePack.ToggleInstantRun" },
        { combo: { key: "l" }, commandId: "Comfy.Canvas.ToggleLinkVisibility" },
    ],
});
