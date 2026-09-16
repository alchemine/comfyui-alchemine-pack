"""Nodes in AlcheminePack/Everywhere.

A broadcast node for cg-use-everywhere with fixed, named inputs.

cg-use-everywhere recognises a node as a broadcaster purely by its type
string -- ``is_UEnode()`` matches ``type.startsWith("Anything Everywhere")``
(js/use_everywhere_utilities.js) -- and all of the rewiring happens in its
frontend, so this node only has to exist and carry the links. It has no
outputs, which means the backend never schedules it.

The reason for a dedicated node rather than the stock "Anything Everywhere":
when one broadcaster carries two inputs of the same type, UE falls back to
matching the *input name* against the target's input name
(``repeated_type_rule`` 0, exact match). On the stock node both conditioning
inputs end up labelled "CONDITIONING" and match nothing, so they have to be
renamed by hand every time. Here ``positive`` and ``negative`` are the names
the schema declares, so they line up with KSampler/SamplerCustom as-is.
"""

from comfy_api.latest import io


class AnythingEverywhereExtended(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            # Must start with "Anything Everywhere" for cg-use-everywhere to
            # pick it up; renaming this breaks the broadcast.
            node_id="Anything Everywhere Extended",
            display_name="Everywhere",
            category="AlcheminePack/Everywhere",
            description=(
                "Broadcast the common sampling inputs to every matching input "
                "in the graph. Requires cg-use-everywhere. Conditioning is "
                "routed by input name, so positive and negative stay apart."
            ),
            inputs=[
                io.Model.Input("model", optional=True),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Conditioning.Input("positive", optional=True),
                io.Conditioning.Input("negative", optional=True),
                io.Latent.Input("latent_image", optional=True),
                io.Int.Input("seed", optional=True),
                # force_input keeps this a socket rather than a text widget, so
                # UE has a link to broadcast. STRING is a very common input type,
                # so see the note in web/js/everywhere_extended.js about scoping
                # this one with input_regex.
                io.String.Input("blacklist", optional=True, force_input=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        # Never actually runs: no outputs means nothing depends on it, and the
        # UE frontend strips it from the submitted prompt.
        return io.NodeOutput()
