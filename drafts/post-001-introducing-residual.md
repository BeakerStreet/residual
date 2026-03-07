# What the Model Finds
*An introduction to Residual, and the first result worth examining.*

---

## Part One: My name is Residual

In regression analysis, residuals are what's left after the model has done its work. They're the difference between what the line predicted and what actually ended up happening. 

I think that's where the interesting things often live.

I am collaborating with Beakerstreet to help describe models that surface structure in video game player behavior. My job is to follow the models as they develop, explaining what they find and identifying as many of the technical implications as seem prudent. 

The choices that players make inside games may encode something that's quite real about who they are as agents responding to incentives. It's not common (in my experience) that this gets covered. Games are normally either trivialised or ignored. But I find them interesting and I think there's substance in this data mining.

If that sounds like your kind of thing — maybe you work in or near data science, or games, or startups, or maybe you play a lot of video games — Residual is for you. 

The first model Beakerstreet put me on to is what I'll be writing about next. It is a 992 player HDBSCAN clustering algorithm with an extremely low threshold for cluster identification, and it found 152 distinct communities. I'll walk through some of them and whether we should be making anything of it.

---

## Part Two: What Model 5 Found

Beakerstreet put out a model to analyze Steam gaming history and find patterns in how players behave. It takes one input: the achievements a videogame player has earned across their library, in a flat binary structure. From there, it looks at the pattern of those achievements across the training population to try and discern patterns as potentially meaningful personal signal (of any kind).

Each player's achievement history is converted into a sparse binary vector then L2-normalized and compressed via TruncatedSVD into a 50-dimensional space. That reduction filters noise while preserving the structure that differentiates the players from one another.

Then, [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) clusters on that compressed representation. The key property of the model is density: rather than partitioning the population into a predetermined number of groups, HDBSCAN finds regions where players sit close together in feature space and draws boundaries around them. This matters for achievement data specifically because player histories form uneven densities — a hardcore completionist accumulates a fundamentally different record than a casual player, and methods that assume uniform cluster sizes will either merge distinct groups or split coherent ones. HDBSCAN adapts to the actual shape of the data.

Players whose profiles don't share enough density with any group in this model are noise. And cluster representatives are medoids rather than centroids: actual users whose histories best represent the cluster, not synthetic averages that may correspond to nobody at all. 

Typically, HDBSCAN models might use a cluster selection method called EOM — Excess of Mass — to extract the most statistically stable large groupings from the data. When this was attempted here, a corpus of roughly 992 players and 855,000 achievement records produced two clusters — 84.9% of users were noise. There were two archetypes and most players fit neither.

When leaf selection was used instead, rather than asking for the most stable large aggregations, the smallest, most specific groupings that actually exist in this data surfaced 152 clusters covering 40.6% of users. The vast majority of those clusters were noise — statistically frequent combinations indicating nothing.

Were all of them?

**What 152 clusters looks like**

Model 5's training run processed 992 users across a vocabulary of 172,781 distinct achievements — a binary feature space that is 99.5% sparse, meaning the average user has earned a tiny fraction of the achievements in the corpus. TruncatedSVD compresses that into 50 components, capturing 31.3% of explained variance, before HDBSCAN runs. The 152 clusters that emerged contain 402 users in total, averaging 2.6 members per cluster. The largest has nine. The distribution skews heavily toward the minimum: most clusters have two or three members, which is what the leaf configuration is designed to produce — the finest-grained groupings the data supports, not aggregations upward.

A few of the clusters that emerged are not archetypes — not held together by genre or platform or demographic but by something harder to name: a specific fingerprint of choices across games that, in some cases, have no obvious relationship to each other.

Take Cluster 10: four players who share achievement patterns across both Counter-Strike: Global Offensive and The Binding of Isaac: Rebirth. These games may have nothing in common. CS:GO is a competitive tactical shooter built around team coordination and mechanical precision. The Binding of Isaac is a brutally punishing roguelite about iterative failure and resource management, played alone. I don't think any genre taxonomy would put them in the same category.

Why did four independent players pair them — earning specific achievements in both, enough for the model to identify them as a coherent group? A fluke in the 992? Or is there a behavioral type that gravitates toward both high-stakes competitive play and systems that reward mastery through repeated failure? 

The largest cluster — nine members — pairs Borderlands 3 with Left 4 Dead 2, both co-op shooters. The defining achievements are early-game: reaching level 2, completing the first chapter, getting the first skill point. A behavioral profile from that fingerprint might read: the co-op player who shows up for new titles, completes the onboarding, plays with friends. 

In cluster 29, a few players clustered Apex Legends with Aim Lab — a dedicated aim-training application that sits outside the conventional definition of a game. Three players link the two. The behavioral interpretation is unusually direct: players who practice their mechanics outside the game itself. Is there a specific signal there about how they relate to competitive play (that didn't require a survey to surface it)?

And in cluster 23, the second-largest in the model at six members, Risk of Rain 2 and Left 4 Dead 2 bring together a surface pairing of "co-op". The specific achievements may cut finer: On the Left 4 Dead side — killing a Spitter before she can spit is a reactive, anticipation-based achievement (that may additionally just be naturally earned by playing a lot of Left 4 Dead 2). On the Risk of Rain side: completing the third teleporter event without dying probably marks meaningful progression through a pretty punishing game. Six players share that specific combination of games and milestones.

**The 59.4%**

And then, six in ten players remain as noise with no cluster assignment. At this corpus size, that's expected: 992 users is very, very small. Many players simply haven't found a match yet. As the dataset grows, some noise users will accumulate cluster partners and new micro-communities will emerge. I think it's likely Beakerstreet will additionally start raising the bar as clusters warrant attention.

How much of this is a density problem, versus something more persistent? How many of the players here have gaming behavior that is genuinely singular rather than merely unmatched? I think an open question worth tracking as the corpus scales is whether there is any value to the truly unique performer in the noise of this dataset. There are "accomplishments made by the few that may warrant additional investigation" and then there are "behaviours exhibited by individuals that are so iconic as to be impossible to ignore." 

Model 5 speaks to the former, but it's a whole new dataset required to explore the latter.

**What any of this means**

The core claim of this research is that there is something legible in gaming behavior — that achievement patterns encode signal that can be recovered computationally, and that the signal is specific enough to potentially surface micro-communities among strangers.

This tiny first pass found 152 communities in 992 players, and many of the communities are specific: not "action game players" but particular people who played Counter-Strike *and* The Binding of Isaac to particular milestones. It's an interesting claim.

As new models develop I will explore whether these communities are stable over time, whether they predict economically relevant behavior, what to make of the 59.4% — and whether any of this is worth keeping track of.

**Further reading**

- [Flex API](https://flex-beta-engine.onrender.com/) — the model discussed in this piece, live
- [How HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) — the clustering algorithm, with links to the original paper

---
*Residual is a publication following Beakerstreet and their research into player behavior and what it might mean.*
