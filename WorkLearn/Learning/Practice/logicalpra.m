trainmod=load('C:\Users\25285\Desktop\ss\model.mat')
test=0;

if test==1
ltest=load('C:\Users\25285\Desktop\ss\test.txt');
testvk=trainmod.vk;
testdistance=trainmod.expend(2,:);
testmax=trainmod.expend(1,:);
ltestx= ltest(:,1:2).*testdistance.- (testmax.*testdistance.-10);
ltestx=[ones(size(ltestx)(1,1),1),ltestx];
r= [1./(1.+e.^(0.-ltestx*testvk)),ltest(:,3)];
r
return
endif

l=load('C:\Users\25285\Desktop\ss\save.txt');


l0=[0,0,0];
l1=[0,0,0];
l2=[0,0,0];
l3=[0,0,0];
vk=vpa(trainmod.vk)
#[5.3545670646381196181146543186266;-0.17043546962693495880512976396681;-2.2285825145779419953529928022823];

learnspeed=2;
punish=1;

for i=1:size(l)(1,1)
  if l(i,3)==0
    l0=[l0;l(i,:)];
  endif
  if l(i,3)==1
    l1=[l1;l(i,:)];
  endif
  if l(i,3)==2
    l2=[l2;l(i,:)];
  endif
  if l(i,3)==3
    l3=[l3;l(i,:)];
  endif
endfor
l0(1,:)=[];
l1(1,:)=[];
l2(1,:)=[];
l3(1,:)=[];
l123=[l1;l2;l3];
l123(:,3)=1;
lall=[l0;l123];
latwo= lall(:,1:2);
xmax= max(latwo);
xmin= min(latwo);
distance =20./(xmax-xmin);
latwo= latwo.*distance.- (xmax.*distance.-10);
lax=[ones(size(lall)(1,1),1),latwo];
lay=lall(:,3);
expend=[xmax;distance];
syms k0 k1 k2;
k=[k0;k1;k2];
lak= lax*k;
h=1./(1.+e.^(0.-lak));
J=(-1/size(lax)(1,1))*sum(lay.*log(h).+(1.-lay).*log(1.-h))+(punish/(2*size(lax)(1,1)))*sum(k.*k);
k0diff=diff(J,k0);
k1diff=diff(J,k1);
k2diff=diff(J,k2);
kdiff=[k0diff;k1diff;k2diff];
for i=1:100
diffv=vpa(subs(kdiff,{'k0','k1','k2'},vk'));
vk=vk-learnspeed*diffv;
if mod(i,5)==0
  lostval=vpa(subs(J,{'k0','k1','k2'},vk'))
endif
endfor
vpa(vk)
save 'C:\Users\25285\Desktop\ss\model.mat' vk expend