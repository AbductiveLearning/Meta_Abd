:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.1479319014448541e-12).
nn('X0',1,5.679091311799889e-12).
nn('X0',2,8.051100476406947e-13).
nn('X0',3,5.50116762310527e-14).
nn('X0',4,8.28213992748678e-15).
nn('X0',5,2.8246383010355203e-10).
nn('X0',6,1.988763721885278e-19).
nn('X0',7,1.0).
nn('X0',8,1.2994614512417102e-18).
nn('X0',9,3.9534388207584925e-08).
nn('X1',0,2.488052359694848e-07).
nn('X1',1,1.371032993802146e-07).
nn('X1',2,0.9999996423721313).
nn('X1',3,3.748037443558269e-14).
nn('X1',4,6.424738949223458e-17).
nn('X1',5,9.551253288709455e-14).
nn('X1',6,7.319866826387134e-13).
nn('X1',7,2.503927898356295e-10).
nn('X1',8,1.0768802516381015e-11).
nn('X1',9,4.286665610013984e-15).
nn('X2',0,1.0).
nn('X2',1,1.1147243926261599e-17).
nn('X2',2,7.50648099234752e-10).
nn('X2',3,7.633758043931784e-17).
nn('X2',4,5.60129035328948e-20).
nn('X2',5,1.2044342303021982e-12).
nn('X2',6,5.268271170873196e-13).
nn('X2',7,3.3881891736697867e-10).
nn('X2',8,7.984497123799533e-14).
nn('X2',9,3.2800202305746304e-14).
nn('X3',0,6.534297991578342e-09).
nn('X3',1,1.589398657131369e-08).
nn('X3',2,8.604930989974946e-09).
nn('X3',3,4.3992215069010854e-05).
nn('X3',4,0.0005783304222859442).
nn('X3',5,4.7475059545831755e-05).
nn('X3',6,5.295356167445142e-11).
nn('X3',7,0.003456357168033719).
nn('X3',8,9.269276546319816e-08).
nn('X3',9,0.995873749256134).

a :- Pos=[f(['X0','X1','X2','X3'],18)], metaabd(Pos).
