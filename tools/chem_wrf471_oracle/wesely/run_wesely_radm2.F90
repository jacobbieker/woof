! RADM2/RADM2SORG layout: chem/module_dep_simple.F:71,1618 with the 42
! gases of Registry/registry.chem:3929-3930 at chem indices 2:43, so
! numgas = p_ho2 = 43 (share/module_chem_share.F get_last_gas).
program run_wesely_radm2
 use module_dep_simple
 use module_state_description
 use module_configure
 use oracle_io
 implicit none
 integer,parameter::nx=24,ny=18,nz=2,ng=43,ngas=42
 type(grid_config_rec_type)::cfg
 real::t(nx,nz,ny),p(nx,nz,ny),pw(nx,nz,ny),tw(nx,nz,ny),rho(nx,nz,ny)
 real::dz(nx,nz,ny),zlev(nx,nz,ny),zw(nx,nz,ny),moist(nx,nz,ny,4),chem(nx,nz,ny,ng)
 real::vel(nx,ny,ng),adef(nx,ny),acen(nx,ny)
 real::skin(nx,ny),sw(nx,ny),veg(nx,ny),pbl(nx,ny),mol(nx,ny),fric(nx,ny)
 real::rough(nx,ny),rain(nx,ny),lat(nx,ny),lon(nx,ny),snow(nx,ny)
 integer::land(nx,ny),i,j,c,day
 real::temp(9)=(/-10.,-5.,-3.,-1.,0.,1.,2.,3.,45./)
 ! dep_init's diffusivities (chem/module_dep_simple.F:1956,2880-2920),
 ! source order; 99. is the :1956 default.  dvj is a dep_init local, so
 ! the oracle checks these against dratio (:3481-3484) instead of dumping it.
 real::dvj_src(ngas)=(/0.126,99.,0.147,0.183,0.175,0.126,0.171,0.151,0.183, &
  0.144,0.127,0.115,0.153,0.124,0.227,0.110,0.127,0.091,0.151,0.118,0.094, &
  0.183,0.189,0.189,0.154,0.121,0.104,0.097,0.115,0.082,0.153,0.113,0.118, &
  0.131,0.118,0.107,0.092,0.096,0.121,99.,0.243,0.174/)
 character(len=48)::case_name,dataset
 num_chem=ng
 ! Registry/registry.chem:3929 order.
 p_so2=2;p_sulf=3;p_no2=4;p_no=5;p_o3=6;p_hno3=7;p_h2o2=8;p_ald=9;p_hcho=10
 p_op1=11;p_op2=12;p_paa=13;p_ora1=14;p_ora2=15;p_nh3=16;p_n2o5=17;p_no3=18
 p_pan=19;p_hc3=20;p_hc5=21;p_hc8=22;p_eth=23;p_co=24;p_ol2=25;p_olt=26
 p_oli=27;p_tol=28;p_xyl=29;p_aco3=30;p_tpan=31;p_hono=32;p_hno4=33;p_ket=34
 p_gly=35;p_mgly=36;p_dcb=37;p_onit=38;p_csl=39;p_iso=40;p_hcl=41;p_ho=42
 p_ho2=43
 do c=1,4
  dataset='USGS'
  cfg%num_land_cat=25
  if(c>=3) then
   dataset='MODIFIED_IGBP_MODIS_NOAH'
   cfg%num_land_cat=20
  endif
  day=180
  if(c==2)day=89
  if(c==4)day=271
  cfg%chem_opt=RADM2
  if(c==2.or.c==3)cfg%chem_opt=RADM2SORG
  if(allocated(luse2usgs))deallocate(luse2usgs)
  call dep_init(1,cfg,ng,trim(dataset),1,nx,1,ny,nx+1,ny+1)
  if(any(0.242/(dvj_src*(293.15/298.15)**1.75)/=dratio(2:ng))) &
   error stop 'dvj_src is not what dep_init used'
  write(case_name,'(a,i2.2)')'radm2/case_',c
  call oracle_open(trim(case_name))
  call oracle_put('julday',day)
  call oracle_put('nrow',ngas)
  call oracle_put('chem_opt',cfg%chem_opt)
  call oracle_put('map_kind',merge(1,0,c>=3))
  call oracle_put('hstar',hstar(2:ng))
  call oracle_put('dhr',dhr(2:ng))
  call oracle_put('f0',f0(2:ng))
  call oracle_put('dvj',dvj_src)
  call oracle_put('dratio',dratio(2:ng))
  call oracle_put('scpr23',scpr23(2:ng))
  t=290.;p=98000.;pw=100000.;tw=290.;rho=1.;dz=50.;zlev=25.
  moist=0.;chem=.01;chem(:,:,:,p_nh3)=.03
  chem(1:nx:2,:,:,p_nh3)=.005
  zw(:,1,:)=0.;zw(:,2,:)=50.
  veg=50.;pbl=500.;rain=0.;lat=35.;lon=0.;snow=0.
  do j=1,ny
   do i=1,nx
    land(i,j)=mod(i-1,cfg%num_land_cat)+1
    skin(i,j)=273.15+temp(mod(j-1,9)+1)
    sw(i,j)=0.
    if(mod(j,3)==1)sw(i,j)=50.
    if(mod(j,3)==2)sw(i,j)=800.
    mol(i,j)=0.
    if(mod(i+j,3)==0)mol(i,j)=.02
    if(mod(i+j,3)==1)mol(i,j)=-.02
    if(i==1)mol(i,j)=5.e-7
    fric(i,j)=.05
    if(mod(j,2)==0)fric(i,j)=1.2
    rough(i,j)=.001
    if(mod(i,2)==0)rough(i,j)=1.
    moist(i,1,j,p_qv)=.001
    if(mod(i+j,8)==0)moist(i,1,j,p_qv)=.1
    if(mod(i+j,8)==1)moist(i,1,j,p_qr)=1.e-5
    if(mod(i+j,8)==2)rain(i,j)=.1
    if(mod(i+j,8)==3)moist(i,1,j,p_qv)=.85*(3.80*exp(17.27*(290.-273.)/(290.-36.))/(.01*98000.))
    if(mod(i+j,8)==4)moist(i,1,j,p_qv)=.80*(3.80*exp(17.27*(290.-273.)/(290.-36.))/(.01*98000.))
    if(i==1)fric(i,j)=1.e-5
    moist(i,1,j,p_qc)=1.e-4
    if(mod(i+j,5)==0)moist(i,1,j,p_qc)=0.
   enddo
  enddo
  vel=-999.;adef=-999.;acen=-999.
  call wesely_driver(1,1,30.,cfg,6,12.,day,t,moist,pw,tw,rain,p,chem,rho,dz, &
  vel,adef,acen,land,skin,sw,veg,pbl,mol,fric,rough,lat,lon,zlev,zw,snow,ng, &
  1,nx+1,1,ny+1,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
  call oracle_put('t_phy_k1',t(:,1,:))
  call oracle_put('p_phy_k1',p(:,1,:))
  call oracle_put('p8w_k1',pw(:,1,:))
  call oracle_put('qv_k1',moist(:,1,:,p_qv))
  call oracle_put('qc_k1',moist(:,1,:,p_qc))
  call oracle_put('qr_k1',moist(:,1,:,p_qr))
  call oracle_put('dz1',zw(:,2,:)-zw(:,1,:))
  call oracle_put('tsk',skin)
  call oracle_put('gsw',sw)
  call oracle_put('vegfra',veg)
  call oracle_put('ust',fric)
  ! Original met, before WRF snaps |rmol|<1e-6 to zero.
  where(abs(mol)<1.e-6)mol=5.e-7
  call oracle_put('rmol',mol)
  call oracle_put('znt',rough)
  call oracle_put('raincv',rain)
  call oracle_put('ivgtyp',land)
  call oracle_put('chem_k1',chem(:,1,:,2:ng))
  call oracle_put('ddvel',vel(:,:,2:ng))
  call oracle_put('aer_res_def',adef)
  call oracle_put('aer_res_zcen',acen)
 enddo
end program
