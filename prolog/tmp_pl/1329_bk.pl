:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,8.1844057084942e-14).
nn('X0',1,8.834366763218365e-16).
nn('X0',2,3.17163694396258e-14).
nn('X0',3,4.9992831055445386e-12).
nn('X0',4,5.817710272944601e-14).
nn('X0',5,1.0).
nn('X0',6,1.7609009953303834e-13).
nn('X0',7,3.517519053808371e-10).
nn('X0',8,5.26669797658026e-15).
nn('X0',9,1.8588042216549638e-09).
nn('X1',0,4.64773464159407e-09).
nn('X1',1,1.0).
nn('X1',2,2.8879573732432107e-10).
nn('X1',3,6.253290348754472e-21).
nn('X1',4,3.065310050058556e-13).
nn('X1',5,2.4126447367711634e-12).
nn('X1',6,8.9324489645215e-13).
nn('X1',7,1.0761216362453752e-09).
nn('X1',8,1.0071485339506817e-13).
nn('X1',9,2.7529877249382606e-13).
nn('X2',0,1.272977279143106e-10).
nn('X2',1,3.727265349660891e-13).
nn('X2',2,2.396420839545499e-10).
nn('X2',3,2.700246795939165e-06).
nn('X2',4,2.7526800749910763e-06).
nn('X2',5,4.4116166719732064e-08).
nn('X2',6,5.054321151605305e-14).
nn('X2',7,0.0018692563753575087).
nn('X2',8,5.085536991522588e-10).
nn('X2',9,0.9981252551078796).
nn('X3',0,9.928802668923709e-09).
nn('X3',1,7.569042992372488e-08).
nn('X3',2,2.0327409799847374e-07).
nn('X3',3,1.0173893315368332e-05).
nn('X3',4,4.641268276373012e-07).
nn('X3',5,0.9999312162399292).
nn('X3',6,1.5506863348946354e-08).
nn('X3',7,1.0225233381788712e-05).
nn('X3',8,1.2679648442315283e-08).
nn('X3',9,4.761049058288336e-05).
nn('X4',0,5.582505764323287e-06).
nn('X4',1,7.686519529670477e-05).
nn('X4',2,0.9998531341552734).
nn('X4',3,9.561576419514495e-09).
nn('X4',4,1.394109972352453e-06).
nn('X4',5,2.4988338509501773e-07).
nn('X4',6,2.81186821666779e-05).
nn('X4',7,8.630708236978535e-08).
nn('X4',8,3.447463677730411e-05).
nn('X4',9,3.1481937057264986e-09).
nn('X5',0,4.219302240926481e-09).
nn('X5',1,1.2441967700544911e-16).
nn('X5',2,1.96253395479129e-11).
nn('X5',3,1.70984405450032e-18).
nn('X5',4,5.48957546087081e-10).
nn('X5',5,3.185501213920361e-07).
nn('X5',6,0.9999996423721313).
nn('X5',7,5.474303528490803e-16).
nn('X5',8,5.0363450795854955e-14).
nn('X5',9,1.0150881717384325e-17).

a :- Pos=[f(['X0','X1'],6),f(['X2','X3','X4','X5'],22)], metaabd(Pos).
