:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.1792706568203304e-11).
nn('X0',1,1.4975275464124138e-09).
nn('X0',2,8.71937022584035e-10).
nn('X0',3,1.1531883087334549e-11).
nn('X0',4,4.111002174529005e-13).
nn('X0',5,1.6316578088648725e-09).
nn('X0',6,1.5956799736603595e-16).
nn('X0',7,0.9999995231628418).
nn('X0',8,1.0717430589852225e-15).
nn('X0',9,4.422284689553635e-07).
nn('X1',0,2.114391290091583e-13).
nn('X1',1,5.02610930652736e-09).
nn('X1',2,2.5900967557390686e-06).
nn('X1',3,0.9999958276748657).
nn('X1',4,5.338607685670361e-15).
nn('X1',5,1.457678649785521e-06).
nn('X1',6,2.0453139244624656e-19).
nn('X1',7,1.650291778787505e-07).
nn('X1',8,2.425553789937862e-13).
nn('X1',9,2.0023915858757846e-10).
nn('X2',0,2.580296219745204e-12).
nn('X2',1,2.958912628803212e-11).
nn('X2',2,1.0037825859399163e-06).
nn('X2',3,8.176111668415548e-20).
nn('X2',4,0.9999927878379822).
nn('X2',5,5.789992428617552e-06).
nn('X2',6,1.0066273148368055e-07).
nn('X2',7,1.772088725759957e-11).
nn('X2',8,2.5484038233066116e-14).
nn('X2',9,4.0930976297204325e-07).
nn('X3',0,6.413862365661771e-07).
nn('X3',1,2.45650971919531e-06).
nn('X3',2,0.0013655239017680287).
nn('X3',3,9.259000233186043e-10).
nn('X3',4,0.9953626394271851).
nn('X3',5,0.0002800359798129648).
nn('X3',6,0.0029731837566941977).
nn('X3',7,2.578697610999825e-08).
nn('X3',8,6.051926249028838e-08).
nn('X3',9,1.5465371689060703e-05).
nn('X4',0,1.6232333255317144e-09).
nn('X4',1,1.0).
nn('X4',2,1.8089885492345026e-11).
nn('X4',3,4.672663712645709e-22).
nn('X4',4,3.313744000336613e-14).
nn('X4',5,7.362503076820126e-13).
nn('X4',6,4.577604910253365e-14).
nn('X4',7,6.481477576869565e-10).
nn('X4',8,1.791623761241562e-15).
nn('X4',9,2.0894859464621468e-14).
nn('X5',0,1.1503214584690567e-15).
nn('X5',1,4.9486508640894655e-12).
nn('X5',2,3.3407427935117084e-09).
nn('X5',3,1.3657489202067996e-11).
nn('X5',4,6.686447185863531e-13).
nn('X5',5,7.743920504843516e-10).
nn('X5',6,3.97770314949919e-12).
nn('X5',7,2.5820875748649996e-07).
nn('X5',8,0.9999995231628418).
nn('X5',9,1.647106131485998e-07).

a :- Pos=[f(['X0','X1','X2','X3'],18),f(['X4','X5'],9)], metaabd(Pos).
