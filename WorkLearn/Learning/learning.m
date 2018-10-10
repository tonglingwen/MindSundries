syms k1 k2 x y lost;
syms py vy;
a=-3.4813624629662863678275495008075;
b=-43.645600041530581957176024972994;
learnspeed=0.001;

datemat=load('C:\Users\25285\Desktop\sym\vrk.txt');
py=datemat(:,2);

y=k1+k2*x;
kmat=[k1;k2];
xmat=py;
xmat(:,2)=xmat(:,1);
xmat(:,1)=1;

for i=1:100
vy=xmat*kmat;
m=size(py)(1,1);
lost=1/(2*m)*sum((vy-py).^2);
vpa(lost);
k1diff=diff(lost,k1);
k2diff=diff(lost,k2);

tema=a-learnspeed* subs(k1diff,{'k1','k2'},[a,b]);
temb=b-learnspeed* subs(k2diff,{'k1','k2'},[a,b]);
a=tema;
b=temb;

lostval=vpa(subs(lost,{'k1','k2'},[a,b]));

vpa(a);
vpa(b);
end