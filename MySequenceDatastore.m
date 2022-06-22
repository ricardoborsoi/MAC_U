classdef MySequenceDatastore < matlab.io.Datastore & ...
                       matlab.io.datastore.MiniBatchable
    
    properties
        Datastore
        SequenceDimension
        MiniBatchSize
    end
    
    properties(SetAccess = protected)
        NumObservations
    end

    properties(Access = private)
        % This property is inherited from Datastore
        CurrentFileIndex
    end


    methods
        
        function ds = MySequenceDatastore(matrix)
            % Construct a MySequenceDatastore object

            % Create a file datastore. The readSequence function is
            % defined following the class definition.
            fds = arrayDatastore(matrix);
            ds.Datastore = fds;
            
            % Determine sequence dimension. When you define the LSTM
            % network architecture, you can use this property to
            % specify the input size of the sequenceInputLayer.
            X = preview(fds);
            ds.SequenceDimension = size(X{1},2);
            ds.NumObservations=size(readall(fds),1);
            % Initialize datastore properties.
            ds.MiniBatchSize = 128;
            ds.CurrentFileIndex = 1;
        end

        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end

        function [dldata,info] = read(ds)            
            % Read one mini-batch batch of data
            miniBatchSize = ds.MiniBatchSize;
            info = struct;
            
            for i = 1:miniBatchSize
                predictors{i,1} = read(ds.Datastore);
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            
            data = preprocessData(ds,predictors,predictors);
            dataarray=nan(size(cell2mat(data{1,1}),2),length(data)*size(cell2mat(data{1,1}),1));
            for i=1:length(data)
                dataarray(:,(i-1)*size(cell2mat(data{1,1}),1)+1:i*size(cell2mat(data{1,1}),1))=cell2mat(data{i,1});
            end
            dldata=dlarray(dataarray,'CB');
        end

        function data = preprocessData(ds,predictors,responses)
            % data = preprocessData(ds,predictors,responses) preprocesses
            % the data in predictors and responses and returns the table
            % data
            
            miniBatchSize = ds.MiniBatchSize;
            
            % Pad data to length of longest sequence.
            sequenceLengths = cellfun(@(X) size(X,2),predictors);
            maxSequenceLength = max(sequenceLengths);
            for i = 1:miniBatchSize
                X = predictors{i};
                
                % Pad sequence with zeros.
                if size(X,2) < maxSequenceLength
                    X(:,maxSequenceLength) = 0;
                end
                
                predictors{i} = X;
            end
            
            % Return data as a table.
            data = predictors;%table(predictors,responses);
        end

        function reset(ds)
            % Reset to the start of the data
            reset(ds.Datastore);
            ds.CurrentFileIndex = 1;
        end
        
        function dsNew = shuffle(ds)
            % dsNew = shuffle(ds) shuffles the files and the
            % corresponding labels in the datastore.
            
            % Create a copy of datastore
            dsNew = copy(ds);
            dsNew.Datastore = shuffle(ds.Datastore);
            fds = dsNew.Datastore;
            
            % Shuffle files and corresponding labels
        end
        
        
    end 

    methods (Hidden = true)

        function frac = progress(ds)
            % Determine percentage of data read from datastore
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end

    end

end % end class definition