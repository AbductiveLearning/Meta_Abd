:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.00012553183478303254).
nn('X0',1,0.006518667098134756).
nn('X0',2,0.9933271408081055).
nn('X0',3,8.913869464777235e-08).
nn('X0',4,1.2264732562883296e-09).
nn('X0',5,3.6163768868391344e-08).
nn('X0',6,2.462711790940375e-06).
nn('X0',7,2.4705110263312235e-05).
nn('X0',8,1.2854378610427375e-06).
nn('X0',9,1.5949948029003735e-09).
nn('X1',0,1.8049353678328828e-11).
nn('X1',1,2.095429829651607e-09).
nn('X1',2,4.3970774754598096e-07).
nn('X1',3,3.7529819666241306e-17).
nn('X1',4,0.9999233484268188).
nn('X1',5,7.106618431862444e-05).
nn('X1',6,9.522396737793315e-08).
nn('X1',7,3.040829588130123e-09).
nn('X1',8,2.253331852705709e-12).
nn('X1',9,5.143306225363631e-06).
nn('X2',0,8.366740633647396e-10).
nn('X2',1,1.0).
nn('X2',2,2.9055920863774176e-11).
nn('X2',3,2.3008231128316308e-23).
nn('X2',4,4.4413047729525285e-14).
nn('X2',5,1.080477531447202e-13).
nn('X2',6,1.931198358509894e-14).
nn('X2',7,2.1472222871271596e-12).
nn('X2',8,3.8495974562637123e-16).
nn('X2',9,1.4566629940630728e-15).
nn('X3',0,6.816595898850863e-13).
nn('X3',1,2.986236535829079e-10).
nn('X3',2,1.0035710040767754e-08).
nn('X3',3,9.083271634846568e-11).
nn('X3',4,2.0958316326868953e-07).
nn('X3',5,3.326598232433753e-07).
nn('X3',6,1.0188135057820347e-11).
nn('X3',7,0.9999984502792358).
nn('X3',8,2.612943932403522e-12).
nn('X3',9,9.306544939136074e-07).

a :- Pos=[f(['X0','X1','X2','X3'],14)], metaabd(Pos).
