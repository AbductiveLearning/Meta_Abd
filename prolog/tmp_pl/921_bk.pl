:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.3227743553279048e-12).
nn('X0',1,3.138051596351943e-08).
nn('X0',2,3.463725306573906e-06).
nn('X0',3,0.9999960660934448).
nn('X0',4,1.4102191981832743e-17).
nn('X0',5,3.916275943538494e-07).
nn('X0',6,2.014326446246522e-19).
nn('X0',7,4.218412771872515e-11).
nn('X0',8,6.506085055332225e-15).
nn('X0',9,2.0135939966738244e-14).
nn('X1',0,5.911558464966049e-10).
nn('X1',1,6.550720286213618e-07).
nn('X1',2,1.7779291738406755e-05).
nn('X1',3,1.4046810292711598e-06).
nn('X1',4,3.7171083988596365e-08).
nn('X1',5,1.2494504062487977e-06).
nn('X1',6,2.401570213805826e-07).
nn('X1',7,0.00011307033128105104).
nn('X1',8,0.9998330473899841).
nn('X1',9,3.259526056353934e-05).
nn('X2',0,1.6769503537530928e-12).
nn('X2',1,1.1198241092009661e-13).
nn('X2',2,2.4634993697758567e-11).
nn('X2',3,4.672230886626494e-07).
nn('X2',4,7.9146684583975e-06).
nn('X2',5,5.8480779330238875e-08).
nn('X2',6,4.115989911098253e-14).
nn('X2',7,0.0012641212670132518).
nn('X2',8,2.2808899657889015e-08).
nn('X2',9,0.9987274408340454).
nn('X3',0,1.4787459180331392e-13).
nn('X3',1,1.5508676955858707e-18).
nn('X3',2,1.2595066258783166e-16).
nn('X3',3,7.340697217327107e-17).
nn('X3',4,1.020642260079552e-14).
nn('X3',5,1.0).
nn('X3',6,2.991375376570904e-12).
nn('X3',7,1.281853719307599e-13).
nn('X3',8,1.5171127708932968e-15).
nn('X3',9,4.026310534976929e-12).
nn('X4',0,0.9999179244041443).
nn('X4',1,3.381945834490807e-10).
nn('X4',2,7.897541945567355e-05).
nn('X4',3,1.8866315454335592e-10).
nn('X4',4,5.448832496313116e-09).
nn('X4',5,3.0902978664926195e-07).
nn('X4',6,2.7697806217474863e-06).
nn('X4',7,1.5964261024237203e-08).
nn('X4',8,1.826754214562243e-08).
nn('X4',9,2.823961064990499e-09).

a :- Pos=[f(['X0','X1'],11),f(['X2','X3','X4'],14)], metaabd(Pos).
