for i=1:length(imdb.images.id)
    if imdb.images.set(i)==1
        folder="training";
    elseif imdb.images.set(i)==2
        folder="validation";
    elseif imdb.images.set(i)==3
        folder="testing";
    end
    filename = [folder,"/",num2str(i),".png"];
    imwrite(imdb.images.data(i), filename)
end 
   
labels = [imdb.images.id, imdb.images.labels];
csvwrite('labels.csv',labels);