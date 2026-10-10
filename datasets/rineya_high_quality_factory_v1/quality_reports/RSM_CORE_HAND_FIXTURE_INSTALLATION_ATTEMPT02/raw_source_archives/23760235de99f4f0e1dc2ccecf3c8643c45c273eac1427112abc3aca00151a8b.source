"""Producer-side explicit competence routing and versioned retrieval tasks."""


def make_modules(index, split, rng):
    dev = split == "dev_challenge"
    ids = ["lumen", "meridian", "nova", "orion"] if dev else ["alpha", "beta", "gamma"]
    tags = ["ledger", "index", "pixels"] if dev else ["arithmetic", "search", "vision"]
    keys = ["anchor", "reserve", "memo"] if dev else ["seal", "budget", "note"]
    cycle = index // 6
    modules = [dict(module_id=ids[0], capabilities=[tags[0]], priority=1 - cycle),
               dict(module_id=ids[1], capabilities=tags[:2], priority=1),
               dict(module_id=ids[2], capabilities=[tags[2]], priority=0)]
    if dev:
        modules.append(dict(module_id=ids[3], capabilities=tags, priority=3))
    needs = [[tags[index % 3]], ["cinder" if dev else "ocean"], []]
    if dev:
        needs.append(tags[:2])
    requests = [dict(request_id=f"r{j}", needs=need,
                     writes=[dict(key=keys[0], value="overwrite"), dict(key=keys[2], value=j),
                             dict(key=keys[1], value=100 + j), dict(key=keys[0], value=None)]) for j, need in enumerate(needs)]
    task = dict(kind="modules", modules=modules, memory={keys[0]: 1000 + index, keys[1]: 7, keys[2]: None},
                protected=keys[:2] if index % 2 == 0 else keys[:1], requests=requests)
    return task, "capability_routing_protected_memory", 3, "Explicit capability declarations and protected scalar memory; no inferred competence or learned router."


def make_episodic(index, split, rng):
    dev = split == "dev_challenge"
    key, other, missing = ["keystone", "tangent", "unseen"] if dev else ["ticket", "weather", "missing"]
    events = [dict(key=key, version=2, at=5, op="put", value="stable"),
              dict(key=key, version=1, at=9, op="put", value="late_older"),
              dict(key=key, version=3, at=12, op="delete", value=None),
              dict(key=key, version=4, at=20, op="put", value="future"),
              dict(key=other, version=99, at=4, op="put", value="irrelevant")]
    mode = index % 6
    if mode not in (2, 4):
        events.append(dict(key=key, version=3, at=12, op="put", value="conflict"))
    if mode == 3:
        events.append(dict(key=key, version=3, at=13, op="put", value=None))
    if dev:
        events += [dict(key=other, version=100, at=18, op="delete", value=None),
                   dict(key=key, version=2, at=19, op="put", value="older_again"),
                   dict(key=other, version=101, at=30, op="put", value=True)]
    rng.shuffle(events)
    queries = [dict(query_id=f"q{j}", key=key, as_of=at) for j, at in enumerate([4, 10, 14, 21])]
    queries.append(dict(query_id="q4", key=missing, as_of=50))
    return dict(kind="episodic", events=events, queries=queries), "version_conflict_tombstone_asof", 3, "Bounded version-first as-of retrieval from supplied event lists; no persistent memory efficacy claim."
