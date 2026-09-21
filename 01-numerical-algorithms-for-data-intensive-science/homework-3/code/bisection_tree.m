function [index_dict,child_dict]=bisection_tree(i1,i2, varargin)
% Does a subdivision of the integers in the interval [i1,i2]
% represented as a binary tree. The optional argument "stopsize" represents
% smallest size when no subdivision should be done.
%
%   [index_dict,child_dict]=bisection_tree(1,101, 5)
%   kv=keys(index_dict) % all nodes
%   node=kv{3}  % node name 3
%   index_dict(node)  % indices for node 3
%   children_nodes=child_dict(node);
%   child1=children_nodes(1,:)
%   child2=children_nodes(2,:)
%
%


    stopsize=1;
    if (length(varargin)>0)
        stopsize=varargin{1};
    end

    something_changed=true;

    node_name='R'; % R as in root
    index_dict=containers.Map();
    index_dict(node_name)=i1:i2;
    child_dict=containers.Map();
    this_level=[node_name];
    while (something_changed)
        next_level=[];
        something_changed=false;
        for i=1:length(this_level)
            node_name=this_level(i,:);
            list=index_dict(node_name);
            j=floor(mean(list));
            if (length(list) > stopsize) % Subdivide
                something_changed=true;

                % Child 1
                child1=[node_name,'a'];
                child1_index=min(list):j;
                index_dict(child1)=child1_index;

                % Child 2
                child2=[node_name,'b'];
                child2_index=(j+1):max(list);
                index_dict(child2)=child2_index;

                % List them as potential nodes for
                % subdivision next round
                next_level=[next_level; child1; child2];

                child_dict(node_name)=[child1;child2];
            end
        end
        this_level=next_level;
    end




end
