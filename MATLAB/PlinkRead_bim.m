function bim = PlinkRead_bim(fileprefix, header, format)

if ~exist('format', 'var'), format = '%s %s %f %d %s %s'; end;
if ~exist('header', 'var'), header = false; end;

bimprefix = [fileprefix,'.bim'];
fprintf('Read in plink bim from %s \r\n', bimprefix);
bimid = fopen(bimprefix,'r');
if header, fgets(bimid); end;
bimdata = textscan(bimid, format);
fclose(bimid);
bim.chrvec = bimdata{1};
bim.snplist = bimdata{2};
bim.cMvec = bimdata{3};
bim.bpvec = bimdata{4};
bim.A1vec = bimdata{5};
bim.A2vec = bimdata{6};

end