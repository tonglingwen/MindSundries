l=load('C:\Users\25285\Desktop\ss\save.txt');
l0=[0,0,0];
l1=[0,0,0];
l2=[0,0,0];
l3=[0,0,0];
vk=[5.6358072983815812536564949540352;-0.47826191175764722045477119527677;-2.0099361534564926179219408157877];
learnspeed=2;
test=0;
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

syms k0 k1 k2;
k=[k0;k1;k2];
lak= lax*k;
h=1./(1.+e.^(0.-lak));
J=(-1/size(lax)(1,1))*sum(lay.*log(h).+(1.-lay).*log(1.-h));
k0diff=diff(J,k0);
k1diff=diff(J,k1);
k2diff=diff(J,k2);
kdiff=[k0diff;k1diff;k2diff];
for i=1:50
diffv=vpa(subs(kdiff,{'k0','k1','k2'},vk'));
vk=vk-learnspeed*diffv;
if mod(i,5)==0
  lostval=vpa(subs(J,{'k0','k1','k2'},vk'))
endif
endfor
vpa(vk)