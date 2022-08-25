:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999998807907104).
nn('X0',1,2.9344148638754675e-14).
nn('X0',2,7.655679468143717e-08).
nn('X0',3,1.0260063333872652e-13).
nn('X0',4,1.627363743978219e-15).
nn('X0',5,7.078485714240301e-11).
nn('X0',6,3.066788489825001e-10).
nn('X0',7,1.8948954405573204e-08).
nn('X0',8,3.106057772095383e-11).
nn('X0',9,1.1195434336530585e-11).
nn('X1',0,1.1755597029150522e-07).
nn('X1',1,0.9999998807907104).
nn('X1',2,2.5373907419634634e-09).
nn('X1',3,2.9960213780152195e-17).
nn('X1',4,3.133353576600939e-10).
nn('X1',5,3.8424294857009045e-09).
nn('X1',6,2.428418022226708e-10).
nn('X1',7,4.4284778510927936e-08).
nn('X1',8,9.729322382467931e-12).
nn('X1',9,5.951537179749167e-11).
nn('X2',0,4.224418148623954e-08).
nn('X2',1,6.069905680305965e-07).
nn('X2',2,3.104025381617248e-06).
nn('X2',3,0.5697967410087585).
nn('X2',4,1.6120281998155406e-06).
nn('X2',5,7.025753438938409e-05).
nn('X2',6,3.078641508391655e-12).
nn('X2',7,0.31417328119277954).
nn('X2',8,3.5724946201298735e-07).
nn('X2',9,0.11595401912927628).
nn('X3',0,1.0).
nn('X3',1,3.936584588388595e-19).
nn('X3',2,2.386545405741458e-10).
nn('X3',3,1.272354193268722e-20).
nn('X3',4,1.724067145636595e-21).
nn('X3',5,2.7493365978894504e-16).
nn('X3',6,8.485603170647893e-14).
nn('X3',7,1.1445005845280541e-14).
nn('X3',8,6.538701011878317e-16).
nn('X3',9,2.1523299095633884e-17).
nn('X4',0,2.5019110339030703e-08).
nn('X4',1,1.1463831787807978e-15).
nn('X4',2,1.0515698034563314e-10).
nn('X4',3,1.2645849523565756e-17).
nn('X4',4,7.432879867153019e-10).
nn('X4',5,8.216135029215366e-06).
nn('X4',6,0.999991774559021).
nn('X4',7,3.95966599962762e-15).
nn('X4',8,1.198454530060289e-10).
nn('X4',9,1.559085476891234e-15).
nn('X5',0,5.930367308337736e-12).
nn('X5',1,7.061076723324788e-11).
nn('X5',2,3.9291950769404504e-12).
nn('X5',3,7.34677110814097e-11).
nn('X5',4,1.5475236380974727e-13).
nn('X5',5,1.096572055381273e-09).
nn('X5',6,9.221594550403211e-19).
nn('X5',7,0.9999971389770508).
nn('X5',8,1.3672350130853293e-14).
nn('X5',9,2.8747704163833987e-06).

a :- Pos=[f(['X0','X1','X2','X3'],4),f(['X4','X5'],13)], metaabd(Pos).