:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.3964406636878266e-07).
nn('X0',1,1.735326563435252e-14).
nn('X0',2,4.715697610180314e-09).
nn('X0',3,2.386542857459777e-16).
nn('X0',4,3.877702603460875e-09).
nn('X0',5,6.781505703656876e-07).
nn('X0',6,0.9999990463256836).
nn('X0',7,3.486101652575707e-13).
nn('X0',8,1.2008793265350093e-11).
nn('X0',9,5.132534818702052e-14).
nn('X1',0,2.2123538556684252e-09).
nn('X1',1,8.753151092832923e-09).
nn('X1',2,5.639627786990786e-08).
nn('X1',3,1.0238063907763717e-07).
nn('X1',4,1.0255457073837793e-11).
nn('X1',5,3.302472251220934e-08).
nn('X1',6,9.432162489204574e-15).
nn('X1',7,0.9999739527702332).
nn('X1',8,3.642161659073162e-12).
nn('X1',9,2.5925868612830527e-05).
nn('X2',0,2.156932838800001e-17).
nn('X2',1,2.745400207084664e-13).
nn('X2',2,1.5483628603421622e-16).
nn('X2',3,1.9930635618056934e-18).
nn('X2',4,2.2237672364076276e-21).
nn('X2',5,2.531176697818764e-15).
nn('X2',6,5.931484339031156e-28).
nn('X2',7,1.0).
nn('X2',8,4.6899648838012715e-23).
nn('X2',9,4.856963355809318e-12).
nn('X3',0,1.8685820946107157e-10).
nn('X3',1,7.609639585126615e-10).
nn('X3',2,2.0011622914317684e-10).
nn('X3',3,5.414200199282959e-09).
nn('X3',4,8.468849233000952e-11).
nn('X3',5,1.931774074037662e-09).
nn('X3',6,4.227743602687921e-17).
nn('X3',7,0.9999825358390808).
nn('X3',8,1.2515036749982666e-12).
nn('X3',9,1.755664561642334e-05).
nn('X4',0,6.108662660153641e-07).
nn('X4',1,1.7328436049349888e-13).
nn('X4',2,6.474818459167864e-10).
nn('X4',3,1.320037850220375e-15).
nn('X4',4,2.4874651671780157e-08).
nn('X4',5,0.00031020279857330024).
nn('X4',6,0.9996892213821411).
nn('X4',7,1.0643635088960005e-13).
nn('X4',8,1.4625666233669676e-09).
nn('X4',9,1.9283871916832285e-13).
nn('X5',0,3.4112497814930975e-05).
nn('X5',1,0.000760567607358098).
nn('X5',2,0.9473170638084412).
nn('X5',3,1.2221090400998946e-05).
nn('X5',4,6.868785540348199e-10).
nn('X5',5,9.556950431033329e-08).
nn('X5',6,2.3720609920729885e-09).
nn('X5',7,0.05187300592660904).
nn('X5',8,7.606842586938001e-07).
nn('X5',9,2.1163111796340672e-06).

a :- Pos=[f(['X0','X1','X2'],20),f(['X3','X4','X5'],15)], metaabd(Pos).
