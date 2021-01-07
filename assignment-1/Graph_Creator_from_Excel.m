OPT = "OFAST"
DIR = "C:\Users\Usuario\Documents\Mathematical modelling and computation\High Performance Computation\high-performance-computing-master_\high-performance-computing-master/";
[rdata,header] = xlsread(DIR+OPT+".xlsx");



for i = 1:numel(header)
Varnames{i} = matlab.lang.makeValidName(strcat(header{i}));
myStruct.(Varnames{i}) = (rdata(:,i));
end

figure(1)
semilogx(myStruct.nmk_kbytes,myStruct.nmk_Mflops,'LineWidth', 2)
hold on
semilogx(myStruct.nkm_kbytes,myStruct.nkm_Mflops,'LineWidth', 2)
semilogx(myStruct.mnk_kbytes,myStruct.mnk_Mflops,'LineWidth', 2)
semilogx(myStruct.mkn_kbytes,myStruct.mkn_Mflops,'LineWidth', 2)
semilogx(myStruct.knm_kbytes,myStruct.knm_Mflops,'LineWidth', 2)
semilogx(myStruct.kmn_kbytes,myStruct.kmn_Mflops,'LineWidth', 2)

legend("nmk","nkm","mnk","mkn","knm","kmn",'Location','northwest')
title(" Permutations Mflop/s - kbytes  OPT-" + OPT )
xlabel("kbytes")
ylabel("Mflop/s")

figure(2)

semilogx(myStruct.nat_kbytes,myStruct.nat_Mflops,'LineWidth', 2)
hold on
semilogx(myStruct.lib_kbytes,myStruct.lib_Mflops,'LineWidth', 2)


legend("nat","lib",'Location','northwest')
title(" Mflop/s - kbytes  OPT-" + OPT )
xlabel("kbytes")
ylabel("Mflop/s")


saveas(figure(1),DIR+'PERM_MFLOPS_'+OPT,'jpg')
saveas(figure(2),DIR+'NAT_MFLOPS_'+OPT,'jpg')