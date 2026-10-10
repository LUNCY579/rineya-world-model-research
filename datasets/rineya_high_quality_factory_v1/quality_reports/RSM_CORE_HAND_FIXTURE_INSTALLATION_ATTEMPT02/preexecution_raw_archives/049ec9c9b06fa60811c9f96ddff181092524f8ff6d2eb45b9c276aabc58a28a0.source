"""Fresh V2 producer under frozen public contracts; this module cannot approve."""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

from factory.common import ROOT, atomic, canonical, digest, load_jsonl, record, record_hash, split_for, write_jsonl
from generators.relational import planning_gap as own

BATCH="planning-gap-v2"
SOURCE_ATTEMPT=4
VERSION="rgm-planning-gap-2.0.2"
MANIFEST_VERSION="rgm-planning-controls-2.0.0"
ACTOR="/root/relational_critic_producer"
SEED=20261010
PUBLIC={
 "schemas/SEMANTICS.md":"718cd0da46f9fb2beaf179526a8c089a69c892cba099632aa6b3af3002371bdd",
 "schemas/RGM_PLANNING_CONTROL_PROPERTIES.md":"f591747cc1b6d0199b76311b5b8283666bf219589f534b42ef195c7b81a7be3a",
 "schemas/RGM_PLANNING_MANIFEST_ENCODING_SUPPLEMENT.md":"02613aa2333f401117def89da224f169d564b527bacc554a0b872877aaac42fe",
 "research/curricula/RGM-NEXT-GAP-PROSPECTIVE.md":"6100d43fe1cf83d798bb34799fac70569f54d33f66692150d188581d9c940761",
 "schemas/RGM_PLANNING_CONTROL_PROPERTIES_V2.md":"d59ce14dd66c6ebd644580f7736e714fbceda32dd132d8c9cb5cf61e78a5a258",
 "research/curricula/RGM-PLANNING-GAP-V2-PROSPECTIVE.md":"34a3c89302d4b965ee784cd2f52207b6e9cb0b4299b52b82bf97c675c2d77d2c",
 "schemas/RGM_PLANNING_V2_RENDERER_CONTROL_ENCODING_ADDENDUM.md":"48e1ebbe954944d8b6569866fa102962607c4ecf27a1d7133f1259f815eac603"}
INPUTS={"research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl":"896e6b1b994dad33f2288680d633afc0b50572140d2bd3aaa2169687529caf5d",
        "reviews/input_only/planning-gap-v1.jsonl":"2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654"}
ROOT_RECEIPTS={"research/curricula/RGM-PLANNING-GAP-V2-ROOT_FREEZE.json":"35259b023c0fa70cf4d044a4d7e1a0d5143f26ed5bab5c1f6646ce1ce49402b9",
 "research/curricula/RGM-PLANNING-GAP-V2-RENDERER-ROOT_FREEZE.json":"7759940fcb0412f67f02c620cc40f52216ca7cbfb695e4f2f5690e99b12f2dbd",
 "research/curricula/RGM-PLANNING-GAP-V2-PRODUCER-ROOT_DISPATCH.json":"0b196febdee0d2f14f5aa7a09ed7d5db903c874c4ab220b22a9d82aac6de96cc"}
DEPENDENCIES=("factory/common.py","factory/reference.py","generators/relational/generate.py","generators/relational/planning_gap.py")
A_SIZES=[5,5,5,5,6,6,7,6,6,7,7,7]
DESTINATION_INDICES={207,209,210,212,213,214}
GUARDS=("source_exists","source_clear","destination_exists_or_table","source_destination_distinct","destination_clear_or_table","changed_support")
OMISSIONS={"source_clear_ignored":"source_clear","destination_clear_ignored":"destination_clear_or_table","no_op_allowed":"changed_support"}
SPACE=[]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def equal(a,b):return canonical(a)==canonical(b)
def check_space(path,stage):
    location=Path(path).resolve()
    while not location.exists():location=location.parent
    free=shutil.disk_usage(location).free;SPACE.append(dict(stage=stage,available_bytes=free))
    assert free>=32*1024*1024,"Storage below32MiB; first current outputs must remain preserved"
def write_json(path,value):check_space(path,Path(path).name);atomic(path,canonical(value)+"\n")
def bindings():return [dict(path=path,sha256=value) for path,value in sorted(PUBLIC.items())]
def assert_public():
    assert all(sha(ROOT/path)==value for path,value in {**PUBLIC,**INPUTS,**ROOT_RECEIPTS}.items())
    assert sha(ROOT/"generators/relational/planning_gap.py")=="0ae4e24c841102cef06a9be22df1aafb4c4b77ca3f97be3efec1dc9e123a192a"


def guard_truth(state,action):
    source,dest=action["block"],action["to"]
    return dict(source_exists=source in state,source_clear=source not in state.values(),
        destination_exists_or_table=dest=="TABLE" or dest in state,source_destination_distinct=source!=dest,
        destination_clear_or_table=dest=="TABLE" or dest not in state.values(),changed_support=source in state and state[source]!=dest)


def block_replay(task,omission=None):
    state=dict(task["on"]);steps=[];valid=True
    ignored=OMISSIONS.get(omission)
    for action in task["plan"]:
        guards=guard_truth(state,action)
        legal=all(value for key,value in guards.items() if key!=ignored)
        if legal:state[action["block"]]=action["to"]
        steps.append(dict(legal=legal,after=dict(state)))
        if not legal:valid=False;break
    return dict(steps=steps,after=state,valid=valid,reaches_goal=valid and all(state[b]==d for b,d in task["goal_on"].items()))


def reference_target(task):
    target=own.reference_target(task)
    if task["kind"]=="blocks":
        assert all(equal(block_replay(task)[key],target[key]) for key in ("steps","after","valid","reaches_goal"))
    return target


def changed_clearance(task,target):
    if task["kind"]!="blocks":return None
    initial=task["on"];initial_clear=set(initial)-set(initial.values())
    for index,step in enumerate(target["steps"]):
        if index<1 or not step["legal"]:continue
        action=task["plan"][index]
        if action["block"] not in initial_clear:code="source_not_clear";obj=action["block"]
        elif action["to"]!="TABLE" and action["to"] not in initial_clear:code="destination_not_clear";obj=action["to"]
        else:continue
        obstruction=next(block for block,support in initial.items() if support==obj)
        for j in range(index):
            before=initial if j==0 else target["steps"][j-1]["after"];after=target["steps"][j]["after"]
            if task["plan"][j]["block"]!=obstruction or obj not in before.values() or obj in after.values():continue
            if any(obj in target["steps"][k]["after"].values() for k in range(j,index)):continue
            return dict(action_index=index,precondition=code,enabling_action_index=j,obstructing_block=obstruction)
    return None


def row_analysis(task,target):
    search=own.graph_search(task) if task["kind"]=="graph" else own.block_search(task)
    assert equal(search["plan"],target["shortest_plan"])
    failure,reason=own.first_failure(task,target);properties={};extra={}
    if failure is not None and failure>=1:properties["legal_prefix_then_first_failure"]=dict(first_error_index=failure)
    if target["valid"] and target["reaches_goal"] and len(task["plan"])>search["minimum_moves"]:properties["valid_goal_detour"]=dict(minimum_moves=search["minimum_moves"])
    if task["kind"]=="graph":
        topology=own.graph_search(task,True);cheap,_=own.cheapest_graph(task)
        extra=dict(topology_minimum_moves=topology["minimum_moves"],minimum_required_fuel=cheap,fuel_subtype="infeasible" if search["plan"] is None else "lengthened" if search["minimum_moves"]>topology["minimum_moves"] else "unlengthened",fuel_boundary_margin=task["fuel"]-cheap if cheap is not None else None)
        if search["plan"] is None and topology["plan"] is not None:properties["fuel_blocks_topologically_reachable_goal"]=dict(topology_distance=topology["minimum_moves"],cheapest_fuel=cheap)
        if search["plan"] is not None and search["minimum_moves"]>topology["minimum_moves"]:properties["fuel_lengthens_fresh_minimum_moves"]=dict(minimum_moves=search["minimum_moves"],topology_distance=topology["minimum_moves"])
    else:
        witness=changed_clearance(task,target)
        if witness:properties["later_action_requires_changed_block_clearance"]=witness
        if 0<len(task["goal_on"])<len(task["on"]):properties["partial_block_goal"]=dict(goal_relation_count=len(task["goal_on"]),block_count=len(task["on"]))
        constrained=own.block_search(task,True)
        if constrained["protected_initial_goal_relations"] and search["plan"] is not None and constrained["plan"] is None:properties["required_temporary_goal_displacement"]=dict(minimum_moves=search["minimum_moves"])
        first_state=task["on"] if failure==0 else target["steps"][failure-1]["after"] if failure is not None else None
        guards=guard_truth(first_state,task["plan"][failure]) if first_state is not None else None
        if failure is not None and failure>=1 and [key for key,value in guards.items() if not value]==["changed_support"] and task["plan"][failure]["to"]=="TABLE":
            properties["isolated_unchanged_support_failure_after_legal_prefix"]=dict(action_index=failure,block=task["plan"][failure]["block"],current_support="TABLE")
        extra=dict(initial_shape=list(own.shape(own.block_encoding(task)[1])),goal_relation_count=len(task["goal_on"]),first_failure_guard_truth=guards,
                   dependency_action_index=witness["action_index"] if witness else None,enabling_action_index=witness["enabling_action_index"] if witness else None,preserving_initial_goal_search=constrained)
    level=4 if search["minimum_moves"] is not None and search["minimum_moves"]>=4 and ("later_action_requires_changed_block_clearance" in properties or "fuel_lengthens_fresh_minimum_moves" in properties) else 3 if properties or (search["minimum_moves"] or 0)>=3 else 2
    if "isolated_unchanged_support_failure_after_legal_prefix" in properties:level=2 if failure==1 else 3
    return properties,dict(kind=task["kind"],valid=target["valid"],reaches_goal=target["reaches_goal"],fresh_plan_feasible=search["plan"] is not None,
        fresh_plan_length=search["minimum_moves"],submitted_plan_length=len(task["plan"]),first_error_index=failure,first_error_reason=reason,
        object_count=len(task["nodes"] if task["kind"]=="graph" else task["on"]),edge_count=len(task["edges"]) if task["kind"]=="graph" else None,
        initial_fuel=task.get("fuel"),search=search,structural_difficulty=level,detour_length=len(task["plan"])-search["minimum_moves"] if target["valid"] and target["reaches_goal"] else None,**extra)


def graph_family(index,ordinal,stratum,rng):
    size=A_SIZES[ordinal] if stratum=="A" else 5+ordinal%3
    subtype="I" if ordinal%2==0 else "L"
    for attempt in range(10000):
        names=rng.sample(own.WORDS,size);start,goal=names[0],names[-1]
        edges={(names[i],names[i+1]):rng.randint(1,3) for i in range(size-1)}
        if stratum=="A" and subtype=="L":edges[start,names[2]]=7;edges[names[2],goal]=7
        if stratum=="B":
            edges={(start,names[1]):rng.randint(1,3),(names[1],names[2]):rng.randint(1,3),(names[2],goal):rng.randint(1,3),(start,names[3]):rng.randint(1,3),(names[3],names[2]):rng.randint(1,3),(names[1],start):rng.randint(1,3)}
            for node in names[4:-1]:edges[names[3],node]=rng.randint(1,3);edges[node,names[2]]=rng.randint(1,3)
        candidates=[(a,b) for a in names for b in names if a!=b and (a,b) not in edges];rng.shuffle(candidates)
        for a,b in candidates[:rng.randint(1,max(2,size+1))]:edges[a,b]=rng.randint(1,7)
        world=dict(kind="graph",nodes=list(names),edges=[[a,b,c] for (a,b),c in edges.items()],start=start,goal=goal,fuel=30,plan=[])
        rng.shuffle(world["nodes"]);rng.shuffle(world["edges"])
        topo=own.graph_search(world,True)
        if stratum=="A" and subtype=="I":
            required,_=own.cheapest_graph(world)
            if required is None or required>30 or topo["minimum_moves"]<3:continue
            base={**world,"fuel":required};plan=own.graph_search(base)["plan"];altered={**base,"fuel":required-1,"plan":plan}
            if own.graph_search(altered)["plan"] is not None:continue
        elif stratum=="A":
            base={**world,"fuel":14};plan=own.graph_search(base)["plan"];required,_=own.cheapest_graph(base)
            altered={**base,"fuel":required,"plan":plan};low=own.graph_search(altered)
            if plan is None or low["plan"] is None or low["minimum_moves"]<=len(plan) or own.reference.graph_run(altered)["valid"]:continue
        else:
            base=world;search=own.graph_search(base);plan=search["plan"]
            if plan is None or len(plan)<3 or (ordinal<6 and search["minimum_route_count"]<2):continue
            altered={**base,"edges":[e for e in world["edges"] if tuple(e[:2])!=(plan[0],plan[1])],"plan":plan}
            if own.graph_search(altered)["plan"] is None:continue
        base={**base,"plan":plan}
        if stratum=="A":
            prefix=None
            for first,cost in sorted((b,c) for a,b,c in base["edges"] if a==start):
                for second,stepcost in sorted((b,c) for a,b,c in base["edges"] if a==first):
                    if second!=goal and cost+stepcost<=base["fuel"]:prefix=[first,second];break
                if prefix:break
            if prefix is None:continue
            missing=next((node for node in sorted(names) if (prefix[0],node) not in edges),None)
            if missing is None:continue
            tasks=[base,altered,{**base,"plan":prefix},{**base,"plan":[prefix[0],missing]}]
        else:
            reverse=next(([b,a] for a,b,c in base["edges"] if a==start and (b,a) not in edges and c<=base["fuel"]),None)
            cost=sum(edges[x,y] for x,y in zip([start]+plan,plan))
            cycle=next(([b,start] for a,b,c in base["edges"] if a==start and (b,start) in edges and c+edges[b,start]+cost<=30),None)
            if reverse is None or cycle is None:continue
            tasks=[base,altered,{**base,"plan":reverse},{**base,"plan":cycle+plan}]
        return tasks,dict(attempts=attempt+1,size=size,A_subtype=subtype if stratum=="A" else None)
    raise RuntimeError(f"Public graph constraints infeasible at{index}; no weakening")


def block_family(index,ordinal,stratum,rng):
    if stratum=="D":return own.block_family(index,ordinal,stratum,rng)
    size=5 if index%2==0 else 6;half=ordinal//2
    proposed=own.SIX_SHAPES[half] if size==6 else own.FIVE_SHAPES[half%6]
    groups=own.states(size);all_states=[s for group in groups.values() for s in group]
    reason="destination_not_clear" if index in DESTINATION_INDICES else "source_not_clear"
    reason_indices=sorted(DESTINATION_INDICES if index in DESTINATION_INDICES else set(range(204,216))-DESTINATION_INDICES)
    preferred_failure=(1 if index in {207,214} else reason_indices.index(index)%3);k=1+ordinal%3
    for attempt in range(6000):
        names=sorted(rng.sample(own.WORDS,size));initial=rng.choice(groups[proposed]);desired=rng.choice(all_states)
        task=own.block_task(initial,dict(enumerate(desired)),names);plan=own.block_search(task)["plan"]
        if not 4<=len(plan)<=8:continue
        base={**task,"plan":plan};target=reference_target(base)
        if not changed_clearance(base,target):continue
        if index in {204,205,206,208} and own.block_search(task,True)["plan"] is not None:continue
        swaps=[]
        for j in range(len(plan)-1):
            variant=copy.deepcopy(plan);variant[j],variant[j+1]=variant[j+1],variant[j]
            changed={**task,"plan":variant};label=reference_target(changed);error,actual_reason=own.first_failure(changed,label)
            if actual_reason!=reason or error!=preferred_failure:continue
            before=initial if error==0 else label["steps"][error-1]["after"]
            if error==0:before=task["on"]
            guards=guard_truth(before,variant[error]);false=[key for key,value in guards.items() if not value]
            expected="source_clear" if reason=="source_not_clear" else "destination_clear_or_table"
            if false==[expected]:swaps.append((j,error,changed,guards))
        if not swaps:continue
        prefix=plan[:k];state=target["steps"][k-1]["after"]
        clear_table=[block for block in sorted(state) if state[block]=="TABLE" and block not in state.values()]
        if not clear_table:continue
        j,error,swapped,guards=swaps[0];block=clear_table[0]
        tasks=[base,swapped,{**task,"plan":own.block_detour(task,plan)},{**task,"plan":prefix},{**task,"plan":prefix+[dict(block=block,to="TABLE")]}]
        return tasks,dict(attempts=attempt+1,size=size,proposed_initial_shape=list(proposed),swap_index=j,first_failure_index=error,
            first_failure_reason=reason,guard_truth=guards,noop_prefix_length=k,noop_block=block)
    raise RuntimeError(f"Public selective C constraints infeasible at{index}; no weakening")


def prior_signatures():
    rows=load_jsonl(ROOT/"research/research_gap_analysis/RGM_PRIOR_ACTIVE_INPUTS.jsonl")
    indices={int(row["source_family_key"].rsplit('-p',1)[1]) for row in rows};assert indices==set(range(180))
    occupied=set();paths=[]
    for directory in ("semantic_parents","candidate_renderings"):
        for path in sorted((ROOT/"data/RGM-001"/directory).glob("*.jsonl")):
            if path.name in ("planning-gap-v1.jsonl",BATCH+".jsonl"):continue
            paths.append(str(path.relative_to(ROOT)))
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if "source_family_index" in row:occupied.add(row["source_family_index"])
                elif "source_family_key" in row:occupied.add(int(row["source_family_key"].rsplit('-p',1)[1]))
    assert not occupied.intersection(range(180,228))
    tasks={digest(row["semantic_task"]):row["semantic_task"] for row in rows}.values()
    return dict(active_indices=sorted(indices),metadata_availability_paths=paths,
        graph_weighted={own.graph_signature(t) for t in tasks if t["kind"]=="graph"},graph_unweighted={own.graph_signature(t,False,False) for t in tasks if t["kind"]=="graph"},
        blocks_joint={own.block_signature(t) for t in tasks if t["kind"]=="blocks"})


def construct():
    prior=prior_signatures();families=[];weighted=set();unweighted=set();blocks=set();sixshapes=set();collisions=Counter()
    for index in range(180,228):
        ordinal=(index-180)%12;stratum="ABCD"[(index-180)//12];rng=random.Random(SEED+index*2027)
        for retry in range(300):
            tasks,construction=graph_family(index,ordinal,stratum,rng) if stratum in "AB" else block_family(index,ordinal,stratum,rng)
            base=tasks[0];signature=own.graph_signature(base) if base["kind"]=="graph" else own.block_signature(base)
            prior_set=prior["graph_weighted"]|weighted if base["kind"]=="graph" else prior["blocks_joint"]|blocks
            if signature in prior_set:collisions[stratum]+=1;continue
            if base["kind"]=="graph":weighted.add(signature);unweighted.add(own.graph_signature(base,False,False))
            else:
                blocks.add(signature)
                if len(base["on"])==6:sixshapes.add(own.shape(own.block_encoding(base)[1]))
            targets=[reference_target(task) for task in tasks];analyzed=[row_analysis(t,g) for t,g in zip(tasks,targets)]
            assert all(len(t["plan"])<=12 for t in tasks) and all(m["search"]["complete"] for _,m in analyzed)
            families.append(dict(index=index,source_family_key=f"RGM-001-seed20261009-p{index:04d}",stratum=stratum,split=split_for(index),tasks=tasks,targets=targets,
                analyzed=analyzed,base_isomorphism_signature=signature,construction=construction));print("FAMILY",index,"attempts",construction["attempts"],flush=True);break
        else:raise RuntimeError(f"Novelty failed index{index}; no gate weakened")
    novel=unweighted-prior["graph_unweighted"]
    assert len(weighted)==len(blocks)==24 and len(novel)>=16 and len(sixshapes)>=8
    assert sum("required_temporary_goal_displacement" in f["analyzed"][0][0] for f in families if f["stratum"]=="C")>=4
    return families,dict(prior_active_families=180,active_indices=prior["active_indices"],reserved_indices_are_revisions_of_v1=True,additional_families_relative_to_v1=0,
        metadata_only_availability_paths=prior["metadata_availability_paths"],new_weighted_graph_terminal_classes=24,new_unweighted_graph_topology_classes=len(novel),
        new_joint_block_initial_goal_classes=24,distinct_six_block_initial_shapes=[list(s) for s in sorted(sixshapes)],collision_rejections=dict(collisions))


def styled_task(task,style):
    out=copy.deepcopy(task) if style=="forward" else own.reorder(copy.deepcopy(task))
    if style=="reverse" and task["kind"]=="graph":out["nodes"].reverse();out["edges"].reverse()
    return out


def build(families):
    rows=[];parents=[];manifest_families=[];measurements=[]
    for family in families:
        index=family["index"];stratum=family["stratum"];contexts=[];sample_ids=[]
        for number,(task,target,(properties,metrics)) in enumerate(zip(family["tasks"],family["targets"],family["analyzed"])):
            context=f"{stratum}{number}";ids=[];witnesses=[]
            for style in ("forward","reverse"):
                payload=styled_task(task,style);assert equal(reference_target(payload),target)
                prompt="Executable task contract:\n"+(own.GRAPH_CONTRACT if task["kind"]=="graph" else own.BLOCK_CONTRACT)+"Task JSON:\n"+json.dumps(payload,ensure_ascii=False,separators=(",",":"))
                row=record("RGM-001",BATCH,index,context,payload,prompt,target,["finite_synthetic_proxy","relational_planning"]+sorted(properties),metrics["structural_difficulty"],"structured",style)
                row.update(generator_version=VERSION,generation_seed=SEED,source_family_key=family["source_family_key"],source_family_index=index,primary_stratum=stratum,
                    controlled_context=context,structural_difficulty=metrics["structural_difficulty"],producer_agent_id=ACTOR,producer_role="relational_planning_producer")
                row["record_hash"]=record_hash(row);rows.append(row);ids.append(row["sample_id"])
                witnesses.extend(dict(sample_id=row["sample_id"],property=name,scope="row",evidence=evidence) for name,evidence in sorted(properties.items()))
                measurements.append(dict(sample_id=row["sample_id"],source_family_key=family["source_family_key"],stratum=stratum,context=context,split=family["split"],renderer=style,row_property_details=properties,**metrics))
            contexts.append(dict(context_id=context,sample_ids=ids,row_properties=sorted(properties),row_witnesses=witnesses));sample_ids.append(ids)
        required=[(0,1,{"A":"fuel_only_counterfactual","B":"directed_edge_deletion_counterfactual","C":"adjacent_action_order_counterfactual","D":"partial_goal_extension_counterfactual"}[stratum],{"A":"fuel","B":"edges","C":"plan","D":"goal_on"}[stratum])]
        if stratum in "BCD":required.append((0,3,"same_world_valid_goal_detour","plan") if stratum=="B" else (0,2,"same_world_valid_goal_detour","plan") if stratum=="C" else (2,3,"same_world_valid_goal_detour","plan"))
        if stratum=="C":required.append((3,4,"isolated_noop_append_counterfactual","plan"))
        controls=[]
        for left,right,name,field in required:
            for si,style in enumerate(("forward","reverse")):
                a,b=styled_task(family["tasks"][left],style),styled_task(family["tasks"][right],style)
                assert [key for key in a if not equal(a[key],b[key])]==[field]
                control=dict(control_id=f"v2-{index}-{name}-{style}",name=name,scope="corpus_variant",sample_ids=[sample_ids[left][si],sample_ids[right][si]],changed_task_fields=[field],provenance="supplied_tasks")
                if name=="adjacent_action_order_counterfactual":control["evidence"]={key:family["construction"][key] for key in ("swap_index","first_failure_index","first_failure_reason","guard_truth")}
                if name=="isolated_noop_append_counterfactual":control["evidence"]=dict(prefix_length=family["construction"]["noop_prefix_length"],block=family["construction"]["noop_block"],current_support="TABLE")
                controls.append(control)
        manifest_families.append(dict(source_family_key=family["source_family_key"],source_index=index,split=family["split"],primary_stratum=stratum,base_sample_id=sample_ids[0][0],contexts=contexts,controls=controls))
        parent=f"RGM-001-{BATCH}-p{index:04d}"
        parents.append(dict(semantic_parent_id=parent,semantic_family_id=parent,source_family_key=family["source_family_key"],source_family_index=index,split=family["split"],primary_stratum=stratum,
            tasks=family["tasks"],contexts=len(family["tasks"]),renderers=2,source_family_count=1,revision_of_batch="planning-gap-v1",additional_source_family_count=0,
            base_isomorphism_signature=family["base_isomorphism_signature"],construction=family["construction"]))
    manifest=dict(version=MANIFEST_VERSION,track_id="RGM-001",batch_name=BATCH,semantics_sha256=PUBLIC["schemas/SEMANTICS.md"],families=manifest_families,
        revision_of_batch="planning-gap-v1",parent_input_view_sha256=INPUTS["reviews/input_only/planning-gap-v1.jsonl"],public_specification_bindings=bindings())
    assert len(rows)==408 and len(parents)==48 and sum(len(f["controls"]) for f in manifest_families)==192
    return rows,parents,manifest,measurements


def complete_block_output(task,replay,shortest):return dict(**replay,shortest_plan=shortest)


def first_difference(strict,mutant):
    for i in range(max(len(strict["steps"]),len(mutant["steps"]))):
        if i>=len(strict["steps"]) or i>=len(mutant["steps"]) or not equal(strict["steps"][i],mutant["steps"][i]):return i
    return None


def hard_cases():
    initial=dict(amber="TABLE",birch="amber",cedar="TABLE",dune="TABLE",ember="TABLE")
    examples=[("sole_source_guard",dict(kind="blocks",on=initial,goal_on=dict(amber="cedar"),plan=[dict(block="amber",to="cedar")]),"source_clear_ignored","source_clear"),
              ("sole_destination_guard",dict(kind="blocks",on=initial,goal_on=dict(cedar="amber"),plan=[dict(block="cedar",to="amber")]),"destination_clear_ignored","destination_clear_or_table"),
              ("isolated_table_noop",dict(kind="blocks",on=initial,goal_on=dict(amber="cedar"),plan=[dict(block="birch",to="TABLE"),dict(block="birch",to="TABLE")]),"no_op_allowed","changed_support")]
    outputs=[]
    for name,task,omission,false_guard in examples:
        target=reference_target(task);index=len(target["steps"])-1;before=task["on"] if index==0 else target["steps"][index-1]["after"]
        guards=guard_truth(before,task["plan"][index]);assert set(guards)==set(GUARDS) and all(type(v) is bool for v in guards.values())
        assert [k for k,v in guards.items() if not v]==[false_guard]
        strict=block_replay(task);mutated=block_replay(task,omission)
        assert mutated["steps"][index]["legal"] and not equal(strict,mutated)
        outputs.append(dict(name=name,input=task,input_sha256=digest(task),guard_truth=guards,omitted_guard=OMISSIONS[omission],strict_output=target,
            mutated_output=complete_block_output(task,mutated,target["shortest_plan"]),first_differing_step=first_difference(strict,mutated),passed=True))
    for code,plan in [("source_not_clear",[dict(block="birch",to="TABLE"),dict(block="amber",to="cedar")]),
                      ("destination_not_clear",[dict(block="birch",to="TABLE"),dict(block="cedar",to="amber")])]:
        task=dict(kind="blocks",on=initial,goal_on=dict(ember="TABLE"),plan=plan);witness=changed_clearance(task,reference_target(task))
        assert witness==dict(action_index=1,precondition=code,enabling_action_index=0,obstructing_block="birch")
        outputs.append(dict(name="typed_clearance_"+code,input=task,input_sha256=digest(task),witness=witness,passed=True))
    both={**initial,"dune":"cedar"}
    task=dict(kind="blocks",on=both,goal_on=dict(ember="TABLE"),plan=[dict(block="birch",to="TABLE"),dict(block="dune",to="TABLE"),dict(block="amber",to="cedar")])
    witness=changed_clearance(task,reference_target(task))
    assert witness==dict(action_index=2,precondition="source_not_clear",enabling_action_index=0,obstructing_block="birch")
    outputs.append(dict(name="both_initial_guards_source_priority",input=task,input_sha256=digest(task),witness=witness,passed=True))
    temp=dict(kind="blocks",on=dict(amber="TABLE",birch="amber",cedar="birch",dune="TABLE",ember="TABLE"),goal_on=dict(amber="cedar",birch="amber",cedar="TABLE",dune="TABLE",ember="TABLE"),plan=[])
    a=own.block_search(temp);p=own.block_search(temp,True)
    assert a["minimum_moves"]==4 and p["plan"] is None and p["complete"]
    outputs.append(dict(name="required_temporary_goal_displacement",input=temp,input_sha256=digest(temp),strict_search=a,preserving_search=p,passed=True))
    # Compound-invalid and unknown-name probes are diagnostic failures for these
    # selective controls, never mislabeled as isolated witnesses.
    compound={**examples[2][1],"plan":[dict(block="birch",to="amber")]};g=guard_truth(initial,compound["plan"][0])
    assert not g["destination_clear_or_table"] and not g["changed_support"]
    assert not block_replay(compound,"no_op_allowed")["valid"]
    outputs.append(dict(name="compound_noop_rejected_as_isolated",input=compound,input_sha256=digest(compound),guard_truth=g,passed=True))
    return dict(actual_execution_id=ACTOR,scope="audit_probe",synthetic_hand_cases_not_production_families=True,actual_invocation=True,completed=True,
                actual_cases=len(outputs),all_passed=True,cases=outputs)


def mutation_probes(rows):
    results=[];required=Counter();hist=Counter();noop_isolation=0
    for row in rows:
        if row["controlled_context"] not in ("C1","C4"):continue
        task=row["semantic_task"];target=row["verified_target"];strict=block_replay(task);index=len(strict["steps"])-1
        before=task["on"] if index==0 else strict["steps"][index-1]["after"];guards=guard_truth(before,task["plan"][index])
        outputs={name:complete_block_output(task,block_replay(task,name),target["shortest_plan"]) for name in sorted(OMISSIONS)}
        context=row["controlled_context"]
        if context=="C1":
            false=[key for key,value in guards.items() if not value];assert len(false)==1 and false[0] in ("source_clear","destination_clear_or_table")
            mutant="source_clear_ignored" if false[0]=="source_clear" else "destination_clear_ignored"
            assert outputs[mutant]["steps"][index]["legal"] and not equal(outputs[mutant],target) and first_difference(target,outputs[mutant])==index
            assert outputs[mutant]["steps"][index]["after"]!=target["steps"][index]["after"]
        else:
            mutant="no_op_allowed";assert [key for key,value in guards.items() if not value]==["changed_support"]
            assert outputs[mutant]["steps"][index]["legal"] and outputs[mutant]["valid"] and not target["valid"] and first_difference(target,outputs[mutant])==index
            assert equal(outputs["source_clear_ignored"],target) and equal(outputs["destination_clear_ignored"],target);noop_isolation+=1
        required[mutant]+=1;hist[canonical([context,mutant,len(task["on"]),row["split"],row["renderer"],index])]+=1
        results.append(dict(scope="audit_probe",sample_id=row["sample_id"],source_family_key=row["source_family_key"],context=context,renderer=row["renderer"],split=row["split"],
            input_sha256=digest(task),actual_invocation=True,completed=True,first_failure_index=index,guard_truth=guards,required_mutant=mutant,
            first_differing_step=first_difference(target,outputs[mutant]),named_omission_action_legal=True,named_omission_transition_executed=True,strict_output=target,mutated_outputs=outputs))
    assert dict(required)=={"source_clear_ignored":12,"destination_clear_ignored":12,"no_op_allowed":24} and noop_isolation==24
    return dict(scope="audit_probe",actual_invocation=True,completed=True,affected_rendered_rows=dict(sorted(required.items())),
        affected_semantic_inputs={name:count//2 for name,count in sorted(required.items())},all_C4_clearance_omissions_agree=24,
        conditional_context_mutant_size_split_style_failure_histogram=dict(sorted(hist.items())),results=results)


def family_checks(families,manifest,rows):
    # Existing producer-only public-control probes cover each original obligation,
    # complete lexical search, read-back, names and illegal suffixes for all204 tasks.
    probes=own.producer_probes(families)
    sources=Counter();destinations=Counter();c4=Counter();firsts=defaultdict(set);subtypes=Counter();counts=Counter()
    by_id={row["sample_id"]:row for row in rows};ids=[]
    for family,entry in zip(families,manifest["families"]):
        index=family["index"];stratum=family["stratum"];metrics=[m for _,m in family["analyzed"]]
        if stratum=="A":
            subtype="I" if index%2==0 else "L";m=metrics[1]
            assert m["object_count"]==A_SIZES[index-180]
            assert (not m["fresh_plan_feasible"])==(subtype=="I")
            subtypes[canonical([subtype,m["object_count"],family["split"]])]+=1
        elif stratum=="B":assert metrics[0]["object_count"]==5+(index-192)%3
        elif stratum=="C":
            assert len(family["tasks"])==5
            reason=metrics[1]["first_error_reason"];expected="destination_not_clear" if index in DESTINATION_INDICES else "source_not_clear"
            assert reason==expected and type(family["construction"]["swap_index"]) is int
            firsts[reason].add(metrics[1]["first_error_index"])
            table=destinations if reason=="destination_not_clear" else sources
            table[canonical([metrics[1]["object_count"],family["split"]])]+=1
            k=1+(index-204)%3;tasks=family["tasks"]
            assert tasks[3]["plan"]==tasks[0]["plan"][:k] and len(tasks[4]["plan"])==k+1 and tasks[4]["plan"][:-1]==tasks[3]["plan"]
            assert metrics[4]["first_error_index"]==k and "isolated_unchanged_support_failure_after_legal_prefix" in family["analyzed"][4][0]
            c4[canonical([k,family["split"]])]+=1
        for control in entry["controls"]:
            ids.append(control["control_id"]);assert control["control_id"] and len(control["sample_ids"])==2
            left,right=(by_id[s] for s in control["sample_ids"])
            assert left["renderer"]==right["renderer"] and left["source_family_key"]==right["source_family_key"]==family["source_family_key"] and left["split"]==right["split"]
            a,b=left["semantic_task"],right["semantic_task"]
            assert sorted(key for key in a if not equal(a[key],b[key]))==control["changed_task_fields"]
            counts[canonical([control["name"],left["renderer"]])]+=1
    assert len(ids)==len(set(ids))==192 and len(firsts["source_not_clear"])>=2 and len(firsts["destination_not_clear"])>=2
    assert len(set.union(*firsts.values()))>=3 and all(any(i>=1 for i in values) for values in firsts.values())
    assert sum(sources.values())==sum(destinations.values())==6 and sum(c4.values())==12
    return dict(original_public_probes=probes,actual_source_failure_reason_size_split=dict(sorted(sources.items())),actual_destination_failure_reason_size_split=dict(sorted(destinations.items())),
        actual_A_subtype_size_split=dict(sorted(subtypes.items())),actual_C4_failure_index_split=dict(sorted(c4.items())),
        C1_first_error_indices_by_reason={reason:sorted(values) for reason,values in sorted(firsts.items())},
        required_semantic_control_pairs=96,actual_required_renderer_declarations=192,control_name_renderer_counts=dict(sorted(counts.items())),all_renderer_bindings_and_task_differences_passed=True)


def diagnostics(rows,measurements,novelty,checks,mutations,fixtures):
    report=own.diagnostics(rows,measurements,novelty,checks["original_public_probes"])
    extra=defaultdict(lambda:defaultdict(Counter))
    for row,m in zip(rows,measurements):
        for group in ("overall",f"family:{row['source_family_key']}",f"stratum:{row['primary_stratum']}",f"context:{row['controlled_context']}",f"split:{row['split']}",
                      f"renderer:{row['renderer']}",f"context_split_style:{row['controlled_context']}:{row['split']}:{row['renderer']}"):
            for field in ("object_count","submitted_plan_length","fresh_plan_feasible","fresh_plan_length","valid","reaches_goal","first_error_reason","first_error_index",
                          "dependency_action_index","enabling_action_index","fuel_subtype","fuel_boundary_margin","detour_length"):
                extra[group][field][canonical(m.get(field))]+=1
            extra[group]["public_row_properties"][canonical(sorted(m["row_property_details"]))]+=1
    type_examples=[(canonical(row["semantic_task"]["kind"]),canonical([m["valid"],m["reaches_goal"],m["fresh_plan_feasible"]]),row["split"]) for row,m in zip(rows,measurements)]
    report["input_only_scalar_and_token_lookups"]["task_type"]=own.feature_lookup(type_examples)
    report.update(version=VERSION,revision_of_batch="planning-gap-v1",additional_source_families_relative_to_v1=0,actual_semantic_tasks=204,
        required_context_and_renderer_checks=checks,selective_mutation_execution=mutations,executed_hand_cases=fixtures,
        extended_conditional_histograms={group:{field:dict(sorted(values.items())) for field,values in sorted(fields.items())} for group,fields in sorted(extra.items())})
    report["limitations"]=["V2 revises the same48 pending V1 lineages; zero additional families relative to V1.204 tasks and408 rows are correlated controls/views.",
        "Two existing grammars only.5/6/7 graphs and5/6 blocks overlap across splits, but each graph dev stratum has only2 families and cannot cover all3 sizes.",
        "C4 provides isolated shorter illegal plans; position/count variation and measured mutation kills do not eliminate every token/length/context cue or incorrect implementation.",
        "Finite lookup fit and exposed split transfer are reported with unknown-bucket majority fallback; no trained planning efficacy, sealed OOD, logistics or edit-distance repair.",
        "Common underlying model and public-design/later-adjudicator overlap remain; this producer has no approval role."]
    return report


def paths(output):
    return [output/f"manifests/{BATCH}_SOURCE_ATTEMPT04.py",output/f"manifests/{BATCH}_SOURCE_FREEZE_ATTEMPT04.json",output/f"manifests/{BATCH}_HAND_FIXTURES_ATTEMPT04.json",
        output/f"data/RGM-001/semantic_parents/{BATCH}.jsonl",output/f"data/RGM-001/candidate_renderings/{BATCH}.jsonl",
        output/f"data/RGM-001/manifests/{BATCH}.family_controls.json",output/f"data/RGM-001/manifests/{BATCH}.producer_integrity.json",output/f"manifests/{BATCH}_GENERATION.json"]


def freeze_source():
    assert_public();snapshot,freeze,hand=paths(ROOT)[:3]
    assert (ROOT/f"manifests/{BATCH}_EXECUTION_ATTEMPT03.json").exists(),"Preserve completed first attempt before new freeze"
    assert not any(path.exists() for path in paths(ROOT))
    fixture=hard_cases();source=Path(__file__).read_text()
    check_space(snapshot,"prospective_source_snapshot");atomic(snapshot,source);write_json(hand,fixture)
    value=dict(actual_author=ACTOR,role="relational_planning_producer",batch=BATCH,generator_version=VERSION,manifest_version=MANIFEST_VERSION,
        source_attempt=SOURCE_ATTEMPT,previous_attempt_source_sha256=sha(ROOT/f"manifests/{BATCH}_SOURCE_ATTEMPT03.py"),previous_attempt_execution_receipt=f"manifests/{BATCH}_EXECUTION_ATTEMPT03.json",unexecuted_previous_source_freeze="manifests/planning-gap-v2_SOURCE_FREEZE_ATTEMPT02.json",producer_choice_revision="C207/C214 selective destination failure index1; require temporary displacement at204/205/206/208 instead of firstfour. All public criteria unchanged.",source_sha256=sha(snapshot),world_seed=SEED,lineage_seed=20261009,source_indices=list(range(180,228)),additional_source_families_relative_to_v1=0,
        candidate_absent_before_freeze=not paths(ROOT)[4].exists(),public_specification_bindings=bindings(),public_input_bindings=INPUTS,root_receipt_bindings=ROOT_RECEIPTS,
        dependency_sha256={path:sha(ROOT/path) for path in DEPENDENCIES},hand_fixture_sha256=sha(hand),hand_fixture_actual_cases=fixture["actual_cases"],
        config=dict(rows=408,semantic_tasks=204,source_families=48,train_rows=340,dev_rows=68,A_sizes=A_SIZES,A_subtypes=["I" if i%2==0 else "L" for i in range(12)],
            C_destination_failure_indices=sorted(DESTINATION_INDICES),C_noop_k_rule="1+(source_index-204)%3",renderer_styles=["forward","reverse"],
            renderer_graph_list_reversal=True,required_semantic_controls=96,required_renderer_declarations=192,max_scratch_bytes=150*1024*1024,hash_seeds=["17","971"]),
        independent_checker_source_sha256_from_root_custody_only="69944b8e06fb984947d0d686296d0bb886aecc358ae6f7405d279d0ba5a40fdf",
        private_checker_tests_outputs_or_expectations_read=False,source_freeze_before_actual_production_generation=True,approval_authority=False)
    write_json(freeze,value);print("SOURCE_FROZEN",sha(snapshot),"FREEZE",sha(freeze),"HAND",sha(hand),flush=True)


def generate(output=ROOT,worker=False):
    output=Path(output).resolve();started=time.perf_counter();cpu=time.process_time();assert_public();core=paths(output)
    source_freeze=json.loads(paths(ROOT)[1].read_text());assert source_freeze["source_sha256"]==sha(Path(__file__))==sha(paths(ROOT)[0])
    assert source_freeze["dependency_sha256"]=={path:sha(ROOT/path) for path in DEPENDENCIES}
    assert source_freeze["public_specification_bindings"]==bindings()
    for path in core[3:]:assert not path.exists(),f"Preserved first output exists:{path}"
    if output!=ROOT:
        for source,target in zip(paths(ROOT)[:3],core[:3]):
            assert not target.exists();check_space(target,"replay_frozen_source_copy");atomic(target,source.read_text())
    rows=[];parents=[];status="FAILED_BEFORE_CANDIDATE";failure=None
    execution=output/f"manifests/{BATCH}_EXECUTION_ATTEMPT04.json"
    try:
        fixtures=hard_cases();assert equal(fixtures,json.loads(core[2].read_text()))
        families,novelty=construct();rows,parents,controls,measurements=build(families)
        checks=family_checks(families,controls,rows);mutations=mutation_probes(rows);integrity=diagnostics(rows,measurements,novelty,checks,mutations,fixtures)
        assert Counter(row["split"] for row in rows)=={"train":340,"dev_challenge":68}
        for row in rows:
            assert row["record_hash"]==record_hash(row) and row["review_status"]=="PENDING" and row["split"]==split_for(row["source_family_index"])
            assert equal(json.loads(row["rendered_input"].split("Task JSON:\n")[1]),row["semantic_task"])
            lead=row["rendered_input"].split("Task JSON:\n")[0]
            assert row["sample_id"] not in lead and row["source_family_key"] not in lead and row["controlled_context"] not in lead
        check_space(core[3],"parent_output");write_jsonl(core[3],parents)
        check_space(core[4],"candidate_output");write_jsonl(core[4],rows);write_json(core[5],controls);write_json(core[6],integrity)
        generation=dict(batch=BATCH,track_id="RGM-001",generator_version=VERSION,manifest_version=MANIFEST_VERSION,status="PENDING",actual_producer=ACTOR,world_seed=SEED,lineage_seed=20261009,
            source_indices=list(range(180,228)),additional_source_families_relative_to_v1=0,actual_counts=dict(rows=408,semantic_tasks=204,source_families=48,train_rows=340,dev_rows=68,required_semantic_controls=96,required_renderer_declarations=192),
            source_attempt=SOURCE_ATTEMPT,source_sha256=source_freeze["source_sha256"],source_freeze_sha256=sha(core[1]),public_specification_bindings=bindings(),public_input_bindings=INPUTS,root_receipt_bindings=ROOT_RECEIPTS,
            dependency_sha256=source_freeze["dependency_sha256"],target_derivation="producer factory.reference plus producer finite BFS/replay checks; no independent oracle imported",
            artifact_sha256={str(path.relative_to(output)):sha(path) for path in core[:-1]},deterministic_core_bytes_before_manifest=sum(path.stat().st_size for path in core[:-1]),
            runtime_measurements_are_separate=True,reproduction_command="PYTHONHASHSEED=17 /opt/anaconda3/bin/python3 -m generators.relational.planning_gap_v2_attempt04 --output-root /ABSOLUTE/NEW/EMPTY/ROOT --worker",no_approval_release_or_push=True)
        write_json(core[7],generation);status="PROVISIONAL_PENDING_INDEPENDENT_REVIEW"
        print("CANDIDATE_FROZEN",sha(core[4]),"ROWS",len(rows),"CORE_BYTES",sum(path.stat().st_size for path in core),flush=True)
        if not worker:
            scratch=output/f"reproducibility/{BATCH}/hashseed-971";assert not scratch.exists();check_space(scratch,"before_isolated_replay")
            command=["/opt/anaconda3/bin/python3","-m","generators.relational.planning_gap_v2_attempt04","--output-root",str(scratch),"--worker"]
            child=subprocess.run(command,cwd=ROOT,env={**os.environ,"PYTHONHASHSEED":"971"},capture_output=True,text=True)
            comparisons=[dict(path=str(path.relative_to(output)),main_sha256=sha(path),replay_sha256=sha(scratch/path.relative_to(output)) if (scratch/path.relative_to(output)).exists() else None) for path in core]
            passed=child.returncode==0 and all(item["main_sha256"]==item["replay_sha256"] for item in comparisons)
            size=sum(path.stat().st_size for path in scratch.rglob('*') if path.is_file());assert size<=150*1024*1024
            receipt=output/f"manifests/{BATCH}_REPRODUCTION.json"
            write_json(receipt,dict(actual_execution_id=ACTOR,hash_seeds=[os.environ.get("PYTHONHASHSEED","process_default"),"971"],separate_processes=True,
                full_deterministic_source_candidate_and_manifest_comparisons=comparisons,passed=passed,scratch_bytes=size,max_scratch_bytes=150*1024*1024,
                actual_command=command,worker_stdout=child.stdout,worker_stderr=child.stderr,
                worker_execution=json.loads((scratch/f"manifests/{BATCH}_EXECUTION_ATTEMPT04.json").read_text()) if (scratch/f"manifests/{BATCH}_EXECUTION_ATTEMPT04.json").exists() else None))
            assert passed,"First failed reproduction retained; never silently regenerate"
            shutil.rmtree(scratch)
    except Exception:
        failure=traceback.format_exc();status="FAILED_PRODUCER_RUN_PRESERVED";raise
    finally:
        write_json(execution,dict(actual_execution_id=ACTOR,role="relational_planning_producer",batch=BATCH,source_attempt=SOURCE_ATTEMPT,source_sha256=source_freeze["source_sha256"],source_freeze_sha256=sha(core[1]),
            actual_python_executable=sys.executable,actual_python_version=sys.version,actual_process_id=os.getpid(),hash_seed=os.environ.get("PYTHONHASHSEED","process_default"),worker_reproduction=worker,
            actual_wall_seconds=time.perf_counter()-started,actual_cpu_seconds=time.process_time()-cpu,actual_reference_invocations=own.REFERENCE_INVOCATIONS,
            actual_unique_cached_shortest_searches=len(own.REFERENCE_CACHE),finite_search_costs=dict(sorted(own.SEARCH_COSTS.items())),disk_space_observations=SPACE,
            existing_core_artifact_sha256={str(path.relative_to(output)):sha(path) for path in core if path.exists()},existing_core_bytes=sum(path.stat().st_size for path in core if path.exists()),
            computed_rows=len(rows),computed_source_parents=len(parents),status=status,initial_failure=failure,no_approval_release_or_push=True))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");parser.add_argument("--output-root",default=str(ROOT));parser.add_argument("--worker",action="store_true")
    args=parser.parse_args()
    if args.freeze:
        try:freeze_source()
        except Exception:
            receipt=ROOT/f"manifests/{BATCH}_PREGENERATION_FAILURE_ATTEMPT04.json"
            assert not receipt.exists();write_json(receipt,dict(actual_author=ACTOR,phase="source_freeze_or_own_hand_diagnostics",source_sha256=sha(Path(__file__)),failure=traceback.format_exc(),candidate_exists=paths(ROOT)[4].exists()));raise
    else:print("Produced",len(generate(args.output_root,args.worker)),"PENDING rows",flush=True)
