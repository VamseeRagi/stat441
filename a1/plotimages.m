function plotimages(images, Y ,scale,proportion)
% function plotimages(images, Y, scale, proportion)
%
% images = images, must be in a 3-dimensional matrix (x by y by n)
% for example if X is 64 by 400 and size of each image is 8 by 8, images=reshape(X,8,8,400);
%
% Y = where to plot the image (Y(1,:) by Y(2,:))
%
% proportion = proportion of the data to be ploted (proportion <= 1).
% for example if there are 400 data points proportion = 1, plots
% all 400 data points and proportion = 0.5 plot only 200 data points 
% (i.e. 1th, 3th, 5th, ...)
% Ali Ghodsi 2006

Y=normr(Y);
inc=floor(1/proportion);
% scale = scale of each image wrt to figure size (scale<1)
%scale=10;
xoff=0;
yoff=0;

xlim = get(gca, 'XLim');
ylim = get(gca, 'YLim');


width = (xlim(2) - xlim(1)) * scale;
height = (ylim(2) - ylim(1)) * scale;

colormap(gray);

image_width = size(images,1);
image_height = size(images,2);
n_images = size(images,3);

xs = Y(1,:) + xoff;
ys = Y(2,:) + yoff;
hold on
for counter = 1:inc:n_images
   
	current_image = 1-reshape(images(:,:,counter), [image_width image_height]);
	imagesc( ...
		[xs(counter)		xs(counter)+width], ...
		[ys(counter)+height	ys(counter)], ...
		current_image' ...
	);
end
xlabel ('x')
ylabel ('y')
hold off
