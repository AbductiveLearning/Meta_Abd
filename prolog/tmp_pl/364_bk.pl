:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.8286569043415013e-20).
nn('X0',2,6.021700933028784e-12).
nn('X0',3,9.715438242827047e-22).
nn('X0',4,2.5550810920744686e-22).
nn('X0',5,2.397676893793198e-15).
nn('X0',6,1.9729243195751311e-13).
nn('X0',7,6.046412980351525e-15).
nn('X0',8,9.542685889261156e-16).
nn('X0',9,5.3871480733830713e-17).
nn('X1',0,8.593477596186005e-10).
nn('X1',1,1.0).
nn('X1',2,2.011774635768404e-10).
nn('X1',3,3.90012042680363e-23).
nn('X1',4,6.36016792617683e-13).
nn('X1',5,2.2877973658314704e-14).
nn('X1',6,5.970637836406629e-15).
nn('X1',7,5.8348300149835275e-12).
nn('X1',8,2.0582873089708712e-15).
nn('X1',9,2.503167012318959e-15).
nn('X2',0,5.381213696864506e-09).
nn('X2',1,2.713910021157062e-07).
nn('X2',2,0.00037668657023459673).
nn('X2',3,7.452424094500643e-11).
nn('X2',4,0.9991500377655029).
nn('X2',5,0.0001909145648824051).
nn('X2',6,2.157763674404123e-06).
nn('X2',7,2.4916000256780535e-05).
nn('X2',8,5.227871913149329e-09).
nn('X2',9,0.00025499609182588756).
nn('X3',0,4.378098496802055e-14).
nn('X3',1,5.693913118776095e-10).
nn('X3',2,3.615080146346372e-08).
nn('X3',3,1.8976532845105254e-10).
nn('X3',4,5.4983542024933385e-11).
nn('X3',5,3.157835593015079e-08).
nn('X3',6,5.597078911456776e-10).
nn('X3',7,1.3402280956142931e-06).
nn('X3',8,0.9999982118606567).
nn('X3',9,4.2963048940691806e-07).

a :- Pos=[f(['X0','X1','X2','X3'],13)], metaabd(Pos).
