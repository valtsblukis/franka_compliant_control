"""Generic identification math and MuJoCo stiction convention; no robot."""
import numpy as np
import pytest
pytest.importorskip('scipy')
from franka_compliant_control.system_identification import fit_resistance


def synthetic(ndof=3):
    rng=np.random.default_rng(21);t=np.arange(0,16,.02)
    dq=np.column_stack([.03*np.cos(t*(.8+.07*j)) for j in range(ndof)])
    ddq=np.column_stack([-.03*(.8+.07*j)*np.sin(t*(.8+.07*j)) for j in range(ndof)])
    fc=np.linspace(.4,.8,ndof);bias=np.linspace(-.1,.08,ndof);b=np.ones(ndof)*.003
    residual=np.sign(dq)*fc+dq*b+bias+rng.normal(0,.015,dq.shape)
    residual[::31]+=.6  # Robust fitting should not turn sparse outliers into friction.
    trial={'directory':'synthetic','q':np.zeros_like(dq),'dq':dq,'ddq':ddq,'residual':residual,
           'valid':np.ones(len(t),bool),'moving':np.ones(len(t),bool),'velocity_noise_std':np.ones(ndof)*.0001,
           'quality':{'sample_period_s':.02}}
    return trial,fc,bias


def test_generic_joint_count_and_robust_bidirectional_recovery():
    trial,fc,bias=synthetic(3)
    result=fit_resistance([trial],prior_coulomb=[.2]*3,prior_viscous=[.003]*3,bootstrap=8)
    assert result['coulomb']==pytest.approx(fc,abs=.025)
    assert result['joint_bias']==pytest.approx(bias,abs=.02)
    assert all(j['fitted'] and not j['at_bound'] for j in result['joints'])
    assert len(result['joints'][0]['conditional_block_bootstrap_95'])==2


def test_unexcited_and_one_direction_joints_remain_unidentified():
    trial,_,_=synthetic()
    trial['dq'][:,0]=0;trial['dq'][:,1]=abs(trial['dq'][:,1])
    result=fit_resistance([trial],prior_coulomb=[.2]*3,prior_viscous=[.003]*3,bootstrap=0)
    assert result['coulomb'][:2]==[.2,.2]
    assert result['joint_bias'][:2]==[0,0]
    assert not result['joints'][0]['fitted'] and not result['joints'][1]['fitted']


def test_validation_data_cannot_leak_into_fit():
    trial,_,_=synthetic();trial['manifest']={'role':'validation'}
    with pytest.raises(ValueError,match='Held-out'):
        fit_resistance([trial],prior_coulomb=[.2]*3,prior_viscous=[.003]*3)


def test_bounds_are_configurable_and_invalid_bounds_rejected():
    trial,_,_=synthetic()
    with pytest.raises(ValueError,match='bounds'):
        fit_resistance([trial],prior_coulomb=[.2]*3,prior_viscous=[.003]*3,coulomb_upper=.1)
    result=fit_resistance([trial],prior_coulomb=[.2]*3,prior_viscous=[.003]*3,
                          coulomb_upper=[1,2,3],bias_bound=[.5,.5,.5],bootstrap=0)
    assert result['settings']['coulomb_upper_Nm']==[1,2,3]


def test_mujoco_soft_default_is_not_ideal_static_coulomb_friction():
    mj=pytest.importorskip('mujoco')
    def displacement(impedance=None,dt=.001):
        attrs='' if impedance is None else f'solimpfriction="{impedance} {impedance} .001 .5 2" solreffriction=".01 1"'
        m=mj.MjModel.from_xml_string(f'''<mujoco><option timestep="{dt}" gravity="0 0 0"/>
         <worldbody><body><joint type="slide" axis="1 0 0" frictionloss="1" {attrs}/>
         <geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>''')
        d=mj.MjData(m);d.qfrc_applied[0]=.5  # Below the 1 N static friction limit.
        for _ in range(round(1/dt)):mj.mj_step(m,d)
        return abs(d.qpos[0])
    soft=displacement();firm=displacement(.9999);half=displacement(.9999,.0005)
    assert soft>100*firm
    assert firm<1e-5 and abs(half-firm)<1e-6
