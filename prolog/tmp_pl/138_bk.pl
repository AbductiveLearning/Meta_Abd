:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,6.6102754381347495e-09).
nn('X0',1,1.0).
nn('X0',2,9.35331950779883e-11).
nn('X0',3,3.048690166915236e-22).
nn('X0',4,6.98458150582941e-14).
nn('X0',5,2.3733653931046206e-12).
nn('X0',6,2.6556259361681933e-12).
nn('X0',7,3.6156245098589723e-12).
nn('X0',8,5.490011860376752e-15).
nn('X0',9,1.4963274082248e-14).
nn('X1',0,1.2355567957644809e-14).
nn('X1',1,1.2833565590439355e-12).
nn('X1',2,1.9851001471413686e-14).
nn('X1',3,3.204823800715122e-16).
nn('X1',4,5.932781303559486e-18).
nn('X1',5,9.231032821813145e-14).
nn('X1',6,1.2360838228750803e-23).
nn('X1',7,1.0).
nn('X1',8,3.168168631385439e-20).
nn('X1',9,5.005700298710281e-10).
nn('X2',0,8.333686309924815e-06).
nn('X2',1,1.0632467706273019e-07).
nn('X2',2,0.00010815773566719145).
nn('X2',3,1.8061584341921844e-05).
nn('X2',4,3.621093128458597e-05).
nn('X2',5,3.138403553748503e-05).
nn('X2',6,1.9909878119506175e-06).
nn('X2',7,0.0006126545486040413).
nn('X2',8,0.7422363758087158).
nn('X2',9,0.25694671273231506).
nn('X3',0,6.8808673824599805e-16).
nn('X3',1,2.0799697034168396e-11).
nn('X3',2,7.539857627136826e-10).
nn('X3',3,5.535932129374643e-11).
nn('X3',4,6.00565430719513e-12).
nn('X3',5,2.8094930826227937e-09).
nn('X3',6,1.9895484999060686e-12).
nn('X3',7,1.8786188320518704e-06).
nn('X3',8,0.999997615814209).
nn('X3',9,5.285255610942841e-07).
nn('X4',0,0.00043413497041910887).
nn('X4',1,8.068235911196098e-05).
nn('X4',2,0.995184600353241).
nn('X4',3,0.004235015716403723).
nn('X4',4,3.3120445275258703e-10).
nn('X4',5,6.146817668195581e-07).
nn('X4',6,2.869374426950344e-08).
nn('X4',7,5.885365681024268e-05).
nn('X4',8,5.272838734526886e-06).
nn('X4',9,8.453059194835078e-07).
nn('X5',0,2.439865021131027e-08).
nn('X5',1,2.6605137648516575e-08).
nn('X5',2,1.0).
nn('X5',3,2.9253577099770667e-13).
nn('X5',4,5.938886591182595e-21).
nn('X5',5,3.4981523691404704e-16).
nn('X5',6,7.399835003649505e-14).
nn('X5',7,4.62129520848592e-13).
nn('X5',8,1.1238252895558881e-13).
nn('X5',9,6.292588326407802e-19).
nn('X6',0,1.7556837761389943e-08).
nn('X6',1,1.9899734135605423e-16).
nn('X6',2,1.3638361273660138e-10).
nn('X6',3,5.175154689800863e-19).
nn('X6',4,2.8008773078624927e-10).
nn('X6',5,2.309300839442585e-07).
nn('X6',6,0.9999997615814209).
nn('X6',7,4.585412330565563e-16).
nn('X6',8,2.7007858297879717e-13).
nn('X6',9,9.294107394928233e-17).

a :- Pos=[f(['X0','X1'],8),f(['X2','X3','X4','X5','X6'],27)], metaabd(Pos).
