:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.2031491254671511e-11).
nn('X0',1,3.382813673624696e-08).
nn('X0',2,8.949282914727519e-07).
nn('X0',3,2.8979516564220376e-09).
nn('X0',4,2.35282376914725e-10).
nn('X0',5,5.8997148499884133e-08).
nn('X0',6,2.0556345958766542e-08).
nn('X0',7,3.227204615541268e-06).
nn('X0',8,0.999995231628418).
nn('X0',9,5.186150247027399e-07).
nn('X1',0,7.415515108162487e-15).
nn('X1',1,9.94666814162347e-22).
nn('X1',2,4.359247515021167e-19).
nn('X1',3,3.9903378038015267e-17).
nn('X1',4,1.0100155745198912e-19).
nn('X1',5,1.0).
nn('X1',6,2.6095630325818084e-15).
nn('X1',7,1.227405585954816e-15).
nn('X1',8,1.8242614087380623e-19).
nn('X1',9,1.2725467245710761e-15).
nn('X2',0,3.772961179038248e-07).
nn('X2',1,1.6714299977844136e-15).
nn('X2',2,1.8179799887718673e-09).
nn('X2',3,7.108519799617992e-19).
nn('X2',4,2.550799763412215e-06).
nn('X2',5,1.0228832252323627e-05).
nn('X2',6,0.9999868273735046).
nn('X2',7,8.859740958642347e-17).
nn('X2',8,9.783202080480141e-14).
nn('X2',9,1.5224212746045053e-14).
nn('X3',0,1.1853182435572762e-08).
nn('X3',1,1.8281937741448928e-08).
nn('X3',2,5.283841986170046e-08).
nn('X3',3,1.696465545819592e-08).
nn('X3',4,2.3128987614029484e-10).
nn('X3',5,9.834827174870497e-09).
nn('X3',6,2.8082909804029443e-14).
nn('X3',7,0.9998909831047058).
nn('X3',8,4.010030973944367e-09).
nn('X3',9,0.00010897662286879495).

a :- Pos=[f(['X0','X1','X2','X3'],26)], metaabd(Pos).
