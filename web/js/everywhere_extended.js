import { app } from "../../scripts/app.js";

// The "Everywhere" broadcast node (node_id "Anything Everywhere Extended") declares a
// fixed set of named inputs -- model / clip / vae / positive / negative /
// latent_image / seed -- because cg-use-everywhere routes same-typed inputs by
// name, and "positive"/"negative" only line up with a sampler if those names
// survive.
//
// UE otherwise manages its broadcasters' inputs dynamically: fix_unconnected_inputs()
// rewrites every unconnected input to type "*" labelled "anything", then
// remove_excess_input() collapses the leftovers into a single spare slot
// (cg-use-everywhere/js/connections.js), which wipes the named schema as soon as
// the node is placed. fix_inputs() bails out early when
// ue_properties.fixed_inputs is set -- the flag Seed Everywhere uses.
//
// Setting that flag once is not enough: UE's setup_ue_properties_oncreate()
// does `node.properties.ue_properties = {...DEFAULT_PROPERTIES}`, replacing the
// whole object, so whichever extension happens to run second wins. Instead of
// racing it, install an accessor on `properties.ue_properties` that re-applies
// the flag on every assignment -- UE can overwrite the object as often as it
// likes and the pin survives.
const NODE_TYPE = "Anything Everywhere Extended";

// UE's input_changed() relabels a slot the moment something is plugged in --
// to the source output's name, or failing that to the type, which is how
// "positive" and "negative" both became "CONDITIONING". It skips that when the
// slot already carries a label of its own ("leave custom label alone"), so
// stamping label = name up front is what makes the schema names stick, and with
// them UE's name-based routing for the two conditioning slots.
function label_inputs(node) {
    (node.inputs || []).forEach((input) => {
        if (input && input.name) input.label = input.name;
    });
}


// input_changed() runs from UE's own onConnectionsChange handler, so setting
// labels at creation is not enough on its own -- whichever handler runs last
// wins. Re-assert after every connection change, plus once more on the next
// tick so we land after any handler registered ahead of ours.
function watch_connections(node) {
    if (node.__alchemine_ue_watched) return;
    node.__alchemine_ue_watched = true;
    const original = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = original?.apply(this, args);
        label_inputs(this);
        setTimeout(() => label_inputs(this), 0);
        return result;
    };
}


function enforce(store, node) {
    if (!store || typeof store !== "object") store = {};
    store.fixed_inputs = true;
    // Route by input NAME, never by type alone. UE's default is type matching,
    // which for a STRING slot like `blacklist` would land in every unconnected
    // STRING input in the graph. repeated_type_rule 4 makes each slot's own name
    // the regex tested against candidate input names, and apply_to_unrepeated
    // extends that from same-typed slots to all of them
    // (cg-use-everywhere/js/connections.js). Substring semantics are what we
    // want here: `seed` reaches noise_seed, `blacklist` reaches blacklist_tags.
    store.repeated_type_rule = 4;
    store.apply_to_unrepeated = 1;
    // Belt and braces: keep_inputs exempts these slots from to_keep()/is_removable()
    // if a future UE version reaches them by another path.
    store.keep_inputs = (node.inputs || []).map((_, i) => i);
    return store;
}

function pin_inputs(node) {
    if (!node || (node.comfyClass ?? node.type) !== NODE_TYPE) return;
    if (!node.properties) node.properties = {};

    const existing = Object.getOwnPropertyDescriptor(node.properties, "ue_properties");
    if (existing && existing.get && existing.set) {
        // Accessor already installed by an earlier call; just refresh the flag.
        label_inputs(node);
        watch_connections(node);
        node.properties.ue_properties = node.properties.ue_properties;
        return;
    }

    label_inputs(node);
    watch_connections(node);

    let store = enforce(node.properties.ue_properties, node);
    Object.defineProperty(node.properties, "ue_properties", {
        configurable: true,
        enumerable: true,
        get: () => store,
        set: (value) => { store = enforce(value, node); },
    });
}

app.registerExtension({
    name: "alchemine.everywhere_extended",
    async nodeCreated(node) {
        pin_inputs(node);
    },
    async loadedGraphNode(node) {
        // configure() can replace node.properties wholesale, so re-install.
        pin_inputs(node);
    },
});
