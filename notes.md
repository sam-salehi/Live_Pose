
- Theres an overall bug with the frames returned freezing. 


- right vs left can be distinguished with speed of the hand
- A punch has elbow spike above 140 and elevation upto 80 lets say for threshold


- Could use velocity in z axis to find upper cuts. 
- Could use net speed for punch detection





It may be a good idea to train decision tree on BoxIV data based on our metrics.


## Punch metrics:




## Data analysis:
Jab metric consistencies:


  1. elev_at_ext — arm elevation at extension: mean 89.7°, std 5.1°, CV 0.056. Your jabs consistently reach roughly horizontal at
  extension. This is very tight. Hooks and uppercuts should differ here significantly.
  2. reach_peak — peak wrist-to-shoulder distance: mean 0.957, std 0.058, CV 0.060. Jabs extend the arm nearly fully. Hooks and uppercuts
  should have lower reach.
  3. vel_X_frac — lateral velocity fraction: mean 0.857, std 0.056, CV 0.066. Interesting — your jabs consistently have dominant lateral. Doesn't make sense. 
  velocity, which confirms why the velocity-axis classifier kept calling them hooks. This metric is consistent but won't separate jabs
  from hooks.
  Hip and shoulder torsion not being emasured nicely? What if legs are visible.
  4. elbow_peak — peak elbow angle in window: mean 162.6°, std 11.2°, CV 0.069. Jabs extend toward straight. This should separate from
  hooks (which stay bent) but the current classifier uses elbow_at_ext (mean 138.7°, CV 0.083) which is noisier because it samples a
  single frame.
  5. reach_at_ext — reach at the extension frame: mean 0.849, std 0.061, CV 0.072. Also tight and high.