syms k1 k2 x y lost;
syms py vy;
a=-0.068047106664523317915319945457572;
b=-0.98190121598370927040597659002423;
learnspeed=0.001;

datemat=load('C:\Users\25285\Desktop\sym\vrk.txt');

plot(datemat(:,1),datemat(:,2),'o')

py=datemat(:,2);

y=k1+k2*x;
kmat=[k1;k2];
xmat=py;
xmat(:,2)=xmat(:,1);
xmat(:,1)=1;
vy=xmat*kmat;
m=size(py)(1,1);
lost=1/(2*m)*sum((vy-py).^2);
k1diff=diff(lost,k1);
k2diff=diff(lost,k2);


for i=1:20
tema=a-learnspeed* subs(k1diff,{'k1','k2'},[a,b]);
temb=b-learnspeed* subs(k2diff,{'k1','k2'},[a,b]);
a=tema;
b=temb;
lostval=vpa(subs(lost,{'k1','k2'},[a,b]))
hold on
plot([0;30],double([a;a+30*b]))
end
vpa(a)
vpa(b)