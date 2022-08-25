:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,3.569687076727051e-19).
nn('X0',2,3.368656395497105e-11).
nn('X0',3,1.8119578726081603e-19).
nn('X0',4,7.961668931630619e-22).
nn('X0',5,4.13853555646191e-14).
nn('X0',6,1.2537320899334847e-13).
nn('X0',7,5.445649695610434e-14).
nn('X0',8,8.40679014303767e-15).
nn('X0',9,7.277184304661815e-17).
nn('X1',0,1.3130690312834759e-08).
nn('X1',1,7.885493456528904e-16).
nn('X1',2,1.4999680664207204e-10).
nn('X1',3,2.9072544758125615e-19).
nn('X1',4,2.4910354667895263e-08).
nn('X1',5,9.240164217771962e-05).
nn('X1',6,0.9999076724052429).
nn('X1',7,1.2718472341753217e-16).
nn('X1',8,4.630585736901949e-13).
nn('X1',9,1.6630919113307606e-15).
nn('X2',0,5.313686123054051e-12).
nn('X2',1,2.2052509507375362e-07).
nn('X2',2,4.559146418614546e-06).
nn('X2',3,0.9999911785125732).
nn('X2',4,1.5622698178470977e-16).
nn('X2',5,4.077161520399386e-06).
nn('X2',6,4.664978949492324e-18).
nn('X2',7,2.8338729141097474e-09).
nn('X2',8,6.665103039794895e-14).
nn('X2',9,3.6213458447231783e-12).
nn('X3',0,1.0).
nn('X3',1,1.9596011943521672e-15).
nn('X3',2,8.827470421302053e-10).
nn('X3',3,4.284162415896611e-16).
nn('X3',4,2.2084132642669638e-23).
nn('X3',5,3.0520219598123566e-12).
nn('X3',6,2.3539828940266627e-13).
nn('X3',7,9.284568253109526e-11).
nn('X3',8,3.885018256535519e-16).
nn('X3',9,4.830814186648487e-17).
nn('X4',0,0.0009025867329910398).
nn('X4',1,0.0003256603959016502).
nn('X4',2,0.9927964806556702).
nn('X4',3,9.414802480023354e-05).
nn('X4',4,0.002876313403248787).
nn('X4',5,0.001983758294954896).
nn('X4',6,0.0009400564595125616).
nn('X4',7,1.0412106803414645e-06).
nn('X4',8,7.081734656821936e-05).
nn('X4',9,9.136541848420165e-06).

a :- Pos=[f(['X0','X1','X2','X3','X4'],11)], metaabd(Pos).
