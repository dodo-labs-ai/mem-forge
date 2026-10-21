"""The conversation layer shared by every conversation Source.

A conversation connector implements only the provider port
(:mod:`memforge.conversations.port`). An executor runs the port for one
collection request and writes the Collection Manifest
(:mod:`memforge.conversations.manifest`). The conversation layer
(:mod:`memforge.conversations.layer`) owns every conversation rule on the
server: message identity, thread and window Units, membership, proven
intervals, conversation status, absence and retention. A connector's Gene,
a :class:`~memforge.genes.conversation_gene.ConversationGene`, binds a sync
run to the layer.
"""
