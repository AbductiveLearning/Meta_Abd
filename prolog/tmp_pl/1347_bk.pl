:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.0355538765670644e-07).
nn('X0',1,3.3040674363160605e-11).
nn('X0',2,2.0904658981635293e-07).
nn('X0',3,4.2922005468078173e-13).
nn('X0',4,6.929690243850928e-07).
nn('X0',5,1.8155702491640113e-05).
nn('X0',6,0.9999806880950928).
nn('X0',7,4.9651421862861866e-11).
nn('X0',8,1.76043102317891e-09).
nn('X0',9,6.560988731474371e-12).
nn('X1',0,6.249110384715095e-08).
nn('X1',1,5.257195880403742e-05).
nn('X1',2,2.031509666267084e-06).
nn('X1',3,8.424516639848889e-10).
nn('X1',4,2.878230764835621e-09).
nn('X1',5,2.081758943006662e-08).
nn('X1',6,5.817735074069297e-13).
nn('X1',7,0.9999445676803589).
nn('X1',8,7.177469122515712e-12).
nn('X1',9,6.86476880673581e-07).
nn('X2',0,9.03415398045837e-10).
nn('X2',1,1.8241501270962798e-19).
nn('X2',2,2.677513497918206e-14).
nn('X2',3,2.4487009901948805e-22).
nn('X2',4,1.089371973704567e-12).
nn('X2',5,8.837099812808447e-06).
nn('X2',6,0.9999911785125732).
nn('X2',7,1.8910297992715117e-18).
nn('X2',8,6.858455589477813e-14).
nn('X2',9,8.438131467199377e-19).
nn('X3',0,1.4410748008231167e-05).
nn('X3',1,0.9999839067459106).
nn('X3',2,2.189513281791733e-07).
nn('X3',3,2.534934285050583e-12).
nn('X3',4,4.884525992565614e-07).
nn('X3',5,4.896731979897595e-07).
nn('X3',6,7.170574178871902e-08).
nn('X3',7,3.0332816436384746e-07).
nn('X3',8,6.751176506725187e-09).
nn('X3',9,6.020403731099577e-08).
nn('X4',0,5.8158209659557505e-15).
nn('X4',1,1.7431101007048255e-09).
nn('X4',2,1.066787476133868e-07).
nn('X4',3,6.6820109445586695e-09).
nn('X4',4,2.2316670822419837e-07).
nn('X4',5,3.301859976545529e-07).
nn('X4',6,1.9852472096792972e-11).
nn('X4',7,4.050287316204049e-05).
nn('X4',8,0.9995763301849365).
nn('X4',9,0.00038247930933721364).
nn('X5',0,2.4013147594814654e-06).
nn('X5',1,4.315969023144239e-12).
nn('X5',2,1.3517821528807872e-08).
nn('X5',3,1.0818266855255887e-12).
nn('X5',4,1.2052864262201979e-09).
nn('X5',5,1.8868522602133453e-05).
nn('X5',6,0.99997878074646).
nn('X5',7,3.396289915968964e-10).
nn('X5',8,1.5663490060546792e-08).
nn('X5',9,4.2000476989552393e-13).

a :- Pos=[f(['X0','X1','X2','X3'],20),f(['X4','X5'],14)], metaabd(Pos).