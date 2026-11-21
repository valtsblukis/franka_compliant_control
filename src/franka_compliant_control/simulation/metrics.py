"""Experiment-owned motion metrics at the native observation rate."""
import numpy as np


class MotionMetrics:
    def __init__(self, measurement):
        self.last=measurement;self.last_acc=None;self.last_contact_s=-np.inf
        self.joint_accel=0.;self.joint_jerk=0.;self.cart_accel=np.zeros(2);self.cart_jerk=np.zeros(2)
        self.free_joint_accel=0.;self.free_joint_jerk=0.;self.free_cart_accel=np.zeros(2);self.free_cart_jerk=np.zeros(2)

    def update(self,measurement,contact=False):
        dt=measurement.time_s-self.last.time_s
        if dt<=0:return
        acc=np.r_[(np.asarray(measurement.velocities)-self.last.velocities)/dt,
                   (np.asarray(measurement.twist)-self.last.twist)/dt]
        if contact:self.last_contact_s=measurement.time_s
        ja=float(np.max(abs(acc[:7])));ca=np.array([np.linalg.norm(acc[7:10]),np.linalg.norm(acc[10:])])
        self.joint_accel=max(self.joint_accel,ja);self.cart_accel=np.maximum(self.cart_accel,ca)
        free=measurement.time_s-self.last_contact_s>.05
        if free:self.free_joint_accel=max(self.free_joint_accel,ja);self.free_cart_accel=np.maximum(self.free_cart_accel,ca)
        if self.last_acc is not None:
            jerk=(acc-self.last_acc)/dt;jj=float(np.max(abs(jerk[:7])));cj=np.array([np.linalg.norm(jerk[7:10]),np.linalg.norm(jerk[10:])])
            self.joint_jerk=max(self.joint_jerk,jj);self.cart_jerk=np.maximum(self.cart_jerk,cj)
            if free:self.free_joint_jerk=max(self.free_joint_jerk,jj);self.free_cart_jerk=np.maximum(self.free_cart_jerk,cj)
        self.last=measurement;self.last_acc=acc

    def report(self,config):
        def ratios(ja,jj,ca,cj):
            return [ja/config['joint_acceleration'],jj/config['joint_jerk'],ca[0]/config['translation_acceleration'],
                    ca[1]/config['rotation_acceleration'],cj[0]/config['translation_jerk'],cj[1]/config['rotation_jerk']]
        return {'joint_acceleration_rad_s2':self.joint_accel,'joint_jerk_rad_s3':self.joint_jerk,
                'cartesian_acceleration':self.cart_accel.tolist(),'cartesian_jerk':self.cart_jerk.tolist(),
                'all_motion_ratios':ratios(self.joint_accel,self.joint_jerk,self.cart_accel,self.cart_jerk),
                'free_motion_ratios':ratios(self.free_joint_accel,self.free_joint_jerk,self.free_cart_accel,self.free_cart_jerk),
                'free_motion_definition':'No true contact in preceding 50 ms; all-contact metrics retained separately'}
