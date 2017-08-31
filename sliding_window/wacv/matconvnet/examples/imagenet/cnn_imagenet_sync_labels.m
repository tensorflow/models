function imdb = cnn_imagenet_sync_labels(imdb, net)
% CNN_IMAGENET_SYNC_LABELS  Match CNN and database labels
%    A CNN NET and the image database IMDB may use a different label ordering.
%    This function matches classes by name and reorder the labels
%    in IMDB to match NET.

[~,perm] = ismember(imdb.classes.name, net.meta.classes.name);
assert(all(perm ~= 0));

imdb.classes.description = imdb.classes.description(perm) ;
imdb.classes.name = imdb.classes.name(perm) ;
ok = imdb.images.label >  0 ;
iperm(perm) = 1:numel(perm) ;
imdb.images.label(ok) = perm(imdb.images.label(ok)) ;


