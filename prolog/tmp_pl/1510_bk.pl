:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,9.701152521301992e-07).
nn('X0',1,0.00031559361377730966).
nn('X0',2,0.9996803402900696).
nn('X0',3,3.476744170338719e-10).
nn('X0',4,8.02504156877104e-12).
nn('X0',5,9.354104296699361e-11).
nn('X0',6,5.036302752614574e-08).
nn('X0',7,1.689508621893765e-06).
nn('X0',8,1.5499219898629235e-06).
nn('X0',9,4.340266340774335e-11).
nn('X1',0,1.0276936399588547e-15).
nn('X1',1,4.4775707426991055e-15).
nn('X1',2,1.6997070417801297e-07).
nn('X1',3,2.0659249477170564e-23).
nn('X1',4,0.9999997615814209).
nn('X1',5,1.3002362209135754e-07).
nn('X1',6,8.149766195231223e-09).
nn('X1',7,5.174361373400105e-13).
nn('X1',8,2.463538747935234e-17).
nn('X1',9,1.2273537741691598e-08).
nn('X2',0,3.901417073848279e-07).
nn('X2',1,2.4447469826327506e-08).
nn('X2',2,8.665331563406653e-08).
nn('X2',3,2.1011659967484775e-08).
nn('X2',4,3.958969870682116e-12).
nn('X2',5,6.023433343216311e-07).
nn('X2',6,8.490461667065574e-10).
nn('X2',7,0.999998927116394).
nn('X2',8,1.1770311559178986e-13).
nn('X2',9,1.0753240964334054e-08).
nn('X3',0,2.866601744931467e-15).
nn('X3',1,5.922756859608191e-13).
nn('X3',2,1.1445005845280541e-14).
nn('X3',3,5.343392998305874e-16).
nn('X3',4,2.475211325169541e-16).
nn('X3',5,4.232040759499789e-11).
nn('X3',6,4.2041881248985175e-22).
nn('X3',7,1.0).
nn('X3',8,3.3251681578828574e-20).
nn('X3',9,2.924331665710156e-09).

a :- Pos=[f(['X0','X1','X2','X3'],20)], metaabd(Pos).
