import copy
import inspect
import numpy as np
import pytest
import torch
from rcwm.returns import discounted_features,linear_utility
from rcwm.masks import causal_mask,prefix_plan_mask,validate_dag,ObservedEvent,available_observation_mask
from rcwm.data import split_for_parent,validate_open_loop
from rcwm.metrics import candidate_regret,brier_score,paired_parent_bootstrap
from rcwm.codebook import TrainOnlyKMeans
from rcwm.synthetic import transition,rollout,make_dataset
from rcwm.model import FeatureTrajectoryTransformer

torch.set_num_threads(2)


def test_discounted_linear_identity():
    f=np.array([[1,2],[3,4],[5,6]],float);w=np.array([.7,-1.2])
    assert np.allclose(linear_utility(discounted_features(f,.8),w),
                       sum((.8**t)*linear_utility(row,w) for t,row in enumerate(f)))


def test_terminal_included_once():
    x=np.ones((4,2))
    assert np.allclose(discounted_features(x,terminated=[False,True,False,False]),[2,2])


def test_truncation_observed_window_only():
    x=np.ones((4,2))
    assert np.allclose(discounted_features(x,truncated=[True,False,False,False]),[1,1])


def test_duration_discount():
    assert np.allclose(discounted_features([[1],[1],[1]],.5,durations=[2,1,4]),[1+.25+.125])


@pytest.mark.parametrize('gamma',[-1,1.01,float('nan')])
def test_invalid_discount(gamma):
    with pytest.raises(ValueError):discounted_features([[1]],gamma)


@pytest.mark.parametrize('x',[[],[1,2],[[float('nan')]]])
def test_invalid_features(x):
    with pytest.raises(ValueError):discounted_features(x)


def test_invalid_duration():
    with pytest.raises(ValueError):discounted_features([[1]],durations=[-1])


def test_non_boolean_termination_rejected():
    with pytest.raises(ValueError):discounted_features([[1]],terminated=[1])


def test_weight_dimension_mismatch():
    with pytest.raises(ValueError):linear_utility([1,2],[1])


def test_goal_revaluation_changes_ranking_not_consequences():
    x=np.array([[1,0,3],[1,1,1]],float);saved=x.copy()
    safe=linear_utility(x,[10,-20,-1]);fast=linear_utility(x,[10,-1,-2])
    assert safe.argmax()==0 and fast.argmax()==1
    assert np.array_equal(x,saved)


def test_coordinate_change_preserves_utility():
    x=np.array([1.,2.]);w=np.array([3.,-2.]);A=np.array([[2.,1.],[0.,3.]])
    assert np.allclose((A@x)@np.linalg.solve(A.T,w),x@w)


def test_causal_mask_semantics():
    m=causal_mask(4)
    assert m.dtype==torch.bool and m[0,3] and not m[3,0] and not m.diagonal().any()


@pytest.mark.parametrize('n',[0,-1,True])
def test_invalid_mask(n):
    with pytest.raises(ValueError):causal_mask(n)


def test_prefix_plan_separation():
    m=prefix_plan_mask(2,3)
    assert m[:2,2:].all() and not m[2:,:].any() and not m[:2,:2].any()


def test_available_observations():
    m=available_observation_mask([1,3],[ObservedEvent(0),ObservedEvent(2),ObservedEvent(0,False)])
    assert m.tolist()==[[False,True,True],[False,False,True]]


@pytest.mark.parametrize('parents',[[[0]],[[],[1]],[[],[0,0]],[[],[-1]]])
def test_bad_dag(parents):
    with pytest.raises(ValueError):validate_dag(parents)


def test_valid_dag():
    validate_dag([[],[],[0,1],[2]])


def record():
    return {'parent_id':'P1','split':'train','plan_origin':'open_loop_precommitted',
            'decision_time':2.,'observations':[{'kind':'observed','available_at':1.,'value':'key'}],
            'actions':[{'operator':'open','executor_version':'v1'}]}


def test_valid_open_loop():validate_open_loop(record())


def test_future_observation_rejected():
    r=record();r['observations'][0]['available_at']=3
    with pytest.raises(ValueError):validate_open_loop(r)


def test_behavior_suffix_rejected():
    r=record();r['plan_origin']='behavior_selected_suffix'
    with pytest.raises(ValueError):validate_open_loop(r)


def test_outcome_in_action_rejected():
    r=record();r['actions'][0]['observed_reward']=1
    with pytest.raises(ValueError):validate_open_loop(r)


def test_group_split_stable():
    assert split_for_parent('parent-one',17)==split_for_parent('parent-one',17)
    assert {split_for_parent(str(i)) for i in range(100)}=={'train','dev','test'}


def test_empty_parent_rejected():
    with pytest.raises(ValueError):split_for_parent('')


def test_regret():
    assert candidate_regret([1,3,2],0)==2 and candidate_regret([1,3,2],1)==0


def test_bad_regret():
    with pytest.raises(ValueError):candidate_regret([1],2)


def test_brier():
    assert brier_score([0,1],[0,1])==0
    assert brier_score([.5,.5],[0,1])==.25


def test_invalid_brier():
    with pytest.raises(ValueError):brier_score([1.1],[1])


def test_bootstrap_parent_weights_not_row_weights():
    ids=['a']*9+['b'];a=[0]*10;b=[1]*9+[0]
    result=paired_parent_bootstrap(ids,a,b,draws=200)
    assert result['difference']==.5 and result['n_parents']==2


def test_bootstrap_fixed_difference():
    result=paired_parent_bootstrap(['a','b','c'],[0,0,0],[1,1,1],draws=100)
    assert result['ci_low']==result['ci_high']==1


def test_bootstrap_one_parent_rejected():
    with pytest.raises(ValueError):paired_parent_bootstrap(['a','a'],[0,0],[1,1])


def test_codebook_train_only():
    with pytest.raises(ValueError):TrainOnlyKMeans(2).fit([[0],[1]],split='test')


def test_codebook_prediction_does_not_refit():
    k=TrainOnlyKMeans(2).fit([[0],[.1],[10],[10.1]],split='train')
    centers=k.centers.copy();pred=k.predict([[.05],[10.05]])
    assert pred[0]!=pred[1] and np.array_equal(centers,k.centers)


def test_codebook_not_fitted():
    with pytest.raises(RuntimeError):TrainOnlyKMeans(2).predict([[1]])


def test_precondition_and_success_once():
    s,f=transition([0,0,0,0],2);assert f[1]==0
    features,states=rollout([0,0,0,0],[1,2,3,3])
    assert states[-1,2]==1 and features[:,2].sum()==1


def test_shortcut_harm_once():
    f,s=rollout([0,0,0,0],[4,4]);assert f[:,3].sum()==1 and s[-1,2]==1


def test_state_snapshot_not_mutated():
    s=np.zeros(4);transition(s,4);assert np.array_equal(s,np.zeros(4))


def test_synthetic_seed_replay():
    a=make_dataset(16,4,9);b=make_dataset(16,4,9)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))


@pytest.fixture
def model():
    torch.manual_seed(3)
    return FeatureTrajectoryTransformer(width=16,heads=4,layers=1,latent=8,max_horizon=4).eval()


def test_model_shapes(model):
    x=model(torch.zeros(2,4),torch.tensor([[1,2,3],[4,0,0]]))
    assert x['features'].shape==(2,3,5) and x['latent'].shape==(2,3,8)


def test_future_actions_do_not_change_earlier_predictions(model):
    s=torch.zeros(1,4)
    with torch.no_grad():
        a=model(s,torch.tensor([[1,2,3,0]]))['features']
        b=model(s,torch.tensor([[1,2,4,4]]))['features']
    torch.testing.assert_close(a[:,:2],b[:,:2],rtol=0,atol=1e-7)


def test_batch_independence(model):
    s=torch.zeros(1,4);a=torch.tensor([[1,2,3]])
    with torch.no_grad():
        alone=model(s,a)['features']
        batch=model(torch.cat([s,torch.ones_like(s)]),torch.cat([a,torch.zeros_like(a)]))['features'][:1]
    torch.testing.assert_close(alone,batch,rtol=1e-5,atol=1e-6)


def test_no_future_target_or_goal_forward_argument(model):
    assert list(inspect.signature(model.forward).parameters)==['observed_initial','candidate_actions']


def test_checkpoint_roundtrip(model,tmp_path):
    p=tmp_path/'m.pt';torch.save(model.state_dict(),p)
    other=copy.deepcopy(model);other.load_state_dict(torch.load(p,weights_only=True))
    with torch.no_grad():
        args=(torch.zeros(1,4),torch.tensor([[1,2]]))
        torch.testing.assert_close(model(*args)['features'],other(*args)['features'],rtol=0,atol=0)


def test_gradient_finite(model):
    x=model(torch.zeros(1,4),torch.tensor([[1,2,3]]))['features'];x.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_bad_action_id(model):
    with pytest.raises(ValueError):model(torch.zeros(1,4),torch.tensor([[10]]))


def test_bad_horizon(model):
    with pytest.raises(ValueError):model(torch.zeros(1,4),torch.zeros(1,5,dtype=torch.long))


def test_nan_state(model):
    with pytest.raises(ValueError):model(torch.full((1,4),float('nan')),torch.tensor([[1]]))
