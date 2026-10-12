#!/usr/bin/env python3
"""Bounded label-free render-control preparation and qualified publication.

prepare writes untrusted metadata only. publish delegates the reviewed locked
backup/snapshot/activation lifecycle; no raw prompts or target cache is created.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from goal_conditioned_decision_data.pipelines import remote_archive as archive
from goal_conditioned_decision_data.pipelines import render_controls as controls


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('pending',help='list missing archived receipt controls; no gold read')
    prepare=commands.add_parser('prepare',help='fully validate at most four raw receipt groups; untrusted staging')
    prepare.add_argument('--receipt',action='append')
    prepare.add_argument('--max-receipts',type=int,default=controls.MAX_NEW_RECEIPTS)
    prepare.add_argument('--seconds',type=float,default=180)
    plan=commands.add_parser('plan',help='aggregate prior qualified controls with untrusted staging')
    plan.add_argument('--capsule',action='append',required=True)
    activate=commands.add_parser('activate',help='rederive new controls and prove already published snapshot')
    activate.add_argument('--bundle',required=True)
    activate.add_argument('--commit',required=True)
    activate.add_argument('--seconds',type=float,default=180)
    publish=commands.add_parser('publish',help='reviewed backup, create-only snapshot, activation and custody backup')
    publish.add_argument('--max-receipts',type=int,default=controls.MAX_NEW_RECEIPTS)
    publish.add_argument('--seconds',type=float,default=180)
    args=parser.parse_args(argv)
    try:
        seconds=getattr(args,'seconds',180)
        if not 0<seconds<=180:raise archive.ArchiveError('invalid control operation deadline')
        deadline=time.monotonic()+seconds
        lock=archive._path(ROOT,'state/worker.lock')
        lock.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:
            with archive._lock(ROOT,fd),archive.read_session(ROOT,deadline=deadline):
                if args.command=='pending':
                    result={'pending_receipts':controls.pending(ROOT),'trusted':False}
                elif args.command=='prepare':
                    result={'capsules':controls.prepare(ROOT,args.receipt,max_receipts=args.max_receipts,deadline=deadline,lock_fd=fd),'trusted':False}
                elif args.command=='plan':
                    result=controls.plan_bundle(ROOT,args.capsule,lock_fd=fd)
                else:
                    archive._check_stop(ROOT)
                    from goal_conditioned_decision_data.scripts import archive_cycle
                    pause=archive._path(ROOT,'state/PAUSED.json')
                    expected_pause=archive._local(ROOT,'state/PAUSED.json',archive.MAX_READ_BYTES)[0] if pause.exists() else None
                    reason=archive._json(expected_pause).get('reason') if expected_pause is not None else None
                    allowed=archive_cycle.DISK_REASONS|{archive_cycle.CLEANUP,archive_cycle.MIGRATION,None,archive_cycle.RENDER_MIGRATION}
                    if reason not in allowed:
                        raise archive.ArchiveError('render control publication cannot override another pause')
                    from goal_conditioned_decision_data.auditors.release import current_independent_review
                    if not current_independent_review(deadline=deadline):
                        raise archive.ArchiveError('render control publication requires current independent source approval')
                    def gate():
                        archive._check_stop(ROOT)
                        current_pause=archive._local(ROOT,'state/PAUSED.json',archive.MAX_READ_BYTES)[0] if pause.exists() else None
                        if current_pause!=expected_pause:
                            raise archive.ArchiveError('pause changed during render control publication')
                        if not current_independent_review(deadline=deadline):
                            raise archive.ArchiveError('review changed during render control publication')
                    if args.command=='activate':
                        result=controls.activate(ROOT,args.bundle,args.commit,deadline=deadline,lock_fd=fd,publication_guard=gate)
                    else:
                        from goal_conditioned_decision_data.scripts.archive_cycle import _publish_render_controls
                        if type(args.max_receipts) is not int or not 1<=args.max_receipts<=controls.MAX_NEW_RECEIPTS:
                            raise archive.ArchiveError('invalid control publication bound')
                        gate()
                        result=_publish_render_controls(ROOT,deadline,archive.read_transport(ROOT,deadline=deadline),fd,
                                                        max_receipts=args.max_receipts)
        finally:os.close(fd)
    except (ValueError,RuntimeError,OSError) as error:
        parser.exit(1,'render controls refused: '+str(error)+'\n')
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
