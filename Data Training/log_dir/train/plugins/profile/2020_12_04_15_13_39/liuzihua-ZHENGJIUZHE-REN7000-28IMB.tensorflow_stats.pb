"�v
BHostIDLE"IDLE1
ףp4�@A
ףp4�@aL�p3W�?iL�p3W�?�Unknown
sHostCast"!functional_1/dropout/dropout/Cast(1�Zd;%q@9�Zd;%q@A�Zd;%q@I�Zd;%q@a��h?G�?i|��)�{�?�Unknown
HostMatMul"+gradient_tape/functional_1/dense_1/MatMul_1(1�G�z�g@9�G�z�g@A�G�z�g@I�G�z�g@a����|�?iQ�BxG�?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1�Vi@9�Vi@A�G�z�f@I�G�z�f@ayC@ʞ\�?im�c8]
�?�Unknown
HostMatMul"+gradient_tape/functional_1/dense_2/MatMul_1(1}?5^�]f@9}?5^�]f@A}?5^�]f@I}?5^�]f@al�rBؗ?i�pw���?�Unknown
}HostMatMul")gradient_tape/functional_1/dense_2/MatMul(1�z�Gid@9�z�Gid@A�z�Gid@I�z�Gid@a���y�?i�0F�1w�?�Unknown
hHostRandomShuffle"RandomShuffle(1!�rh�d@9!�rh�d@A!�rh�d@I!�rh�d@al��S�?i����!�?�Unknown
�HostRandomUniform"9functional_1/dropout/dropout/random_uniform/RandomUniform(1�|?5^�b@9�|?5^�b@A�|?5^�b@I�|?5^�b@a@ȱ��!�?iaG���?�Unknown
s	Host_FusedMatMul"functional_1/dense_1/Relu(1�|?5^F`@9�|?5^F`@A�|?5^F`@I�|?5^F`@a �Sx�Y�?iI�	W�M�?�Unknown
}
HostMatMul")gradient_tape/functional_1/dense_1/MatMul(1�����,Z@9�����,Z@A�����,Z@I�����,Z@a����?i�RP�K��?�Unknown
HostMatMul"+gradient_tape/functional_1/dense_3/MatMul_1(1V-���X@9V-���X@AV-���X@IV-���X@as�S��{�?im���:'�?�Unknown
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1��~j��X@9��~j��X@A��~j��X@I��~j��X@a�ˇ��x�?i������?�Unknown
HostReluGrad")gradient_tape/functional_1/dense/ReluGrad(1��Q�^X@9��Q�^X@A��Q�^X@I��Q�^X@a�JH}���?i���	��?�Unknown
qHost_FusedMatMul"functional_1/dense/Relu(1+��.X@9+��.X@A+��.X@I+��.X@av25��ǉ?i���9)`�?�Unknown
sHost_FusedMatMul"functional_1/dense_2/Relu(1� �rh�W@9� �rh�W@A� �rh�W@I� �rh�W@a���?i�&��d��?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1R���q]@9R���q]@A��Q��W@I��Q��W@a!�m�0W�?i݊��+�?�Unknown
�HostReluGrad"+gradient_tape/functional_1/dense_2/ReluGrad(1L7�A`�V@9L7�A`�V@AL7�A`�V@IL7�A`�V@a#�`�?��?i�_�ʍ�?�Unknown
sHostMul""functional_1/dropout/dropout/Mul_1(1F�����R@9F�����R@AF�����R@IF�����R@a�$2��?i��;)��?�Unknown
}HostMatMul")gradient_tape/functional_1/dense_3/MatMul(1u�V�P@9u�V�P@Au�V�P@Iu�V�P@a��~��ف?i���q�%�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1��� ��O@9��� ��O@A��� ��O@I��� ��O@a�s\v��?i�^���i�?�Unknown
�HostReluGrad"+gradient_tape/functional_1/dense_1/ReluGrad(1)\���hJ@9)\���hJ@A)\���hJ@I)\���hJ@am�/��'|?iO�����?�Unknown
qHostSoftmax"functional_1/dense_3/Softmax(133333#I@933333#I@A33333#I@I33333#I@a�a��f�z?i��̕��?�Unknown
vHost_FusedMatMul"functional_1/dense_3/BiasAdd(1���(\�G@9���(\�G@A���(\�G@I���(\�G@aŻɫ8�y?i�>�
�?�Unknown
{HostMatMul"'gradient_tape/functional_1/dense/MatMul(1w��/�F@9w��/�F@Aw��/�F@Iw��/�F@a]oG�Nx?ijQ��;;�?�Unknown
`HostGatherV2"
GatherV2_1(1��Mb�C@9��Mb�C@A��Mb�C@I��Mb�C@aH��H�t?i��%�e�?�Unknown
�HostGreaterEqual")functional_1/dropout/dropout/GreaterEqual(1��Q��A@9��Q��A@A��Q��A@I��Q��A@a�Y���s?i]g����?�Unknown
^HostGatherV2"GatherV2(1��� ��@@9��� ��@@A��� ��@@I��� ��@@a iD1�q?i]9lYٮ�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1�x�&1�@@9�x�&1�@@A�x�&1�@@I�x�&1�@@a�>N�ԟq?i����?�Unknown
qHostMul" functional_1/dropout/dropout/Mul(1�z�G�>@9�z�G�>@A�z�G�>@I�z�G�>@a�}�>�dp?i�*�����?�Unknown
�HostMul"0gradient_tape/functional_1/dropout/dropout/Mul_1(1�Zd�=@9�Zd�=@A�Zd�=@I�Zd�=@a�'
��o?i�4s��?�Unknown
\HostArgMax"ArgMax_1(1����S=@9����S=@A����S=@I����S=@ȃ(ZDo?i�]�2�?�Unknown
u HostFlushSummaryWriter"FlushSummaryWriter(11�Z�9@91�Z�9@A1�Z�9@I1�Z�9@a�O��3k?i��QM�?�Unknown�
Z!HostArgMax"ArgMax(1�t�68@9�t�68@A�t�68@I�t�68@a�"{���i?i��� g�?�Unknown
�"HostBiasAddGrad"6gradient_tape/functional_1/dense_1/BiasAdd/BiasAddGrad(1�G�z�5@9�G�z�5@A�G�z�5@I�G�z�5@a�y�ag?i�z��~�?�Unknown
�#HostBiasAddGrad"6gradient_tape/functional_1/dense_2/BiasAdd/BiasAddGrad(1�rh���4@9�rh���4@A�rh���4@I�rh���4@a�%�E�f?i��\���?�Unknown
$HostMul".gradient_tape/dense_1/kernel/Regularizer/Mul_1(1sh��|_4@9sh��|_4@Ash��|_4@Ish��|_4@a�o��e?i+l�uU��?�Unknown
�%HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1�v���2@9�v���2@A�v���2@I�v���2@a �M�$�c?i¹֚@��?�Unknown
�&HostBiasAddGrad"4gradient_tape/functional_1/dense/BiasAdd/BiasAddGrad(1;�O���2@9;�O���2@A;�O���2@I;�O���2@a;��C��c?iZ�4+��?�Unknown
�'HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1+��η1@9+��η1@Aj�t��/@Ij�t��/@aդ ��`?i/'���?�Unknown
i(HostWriteSummary"WriteSummary(1/�$�-@9/�$�-@A/�$�-@I/�$�-@a[0��t_?iG¼��?�Unknown�
d)HostDataset"Iterator::Model(1�/�$�2@9�/�$�2@A��v��*@I��v��*@aي��,�[?iqKئ �?�Unknown
�*HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1�"��~�)@9�"��~�)@A��|?5)@I��|?5)@aƬ�q�Z?ib;�b
�?�Unknown
�+HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1j�t��&@9j�t��&@Aj�t��&@Ij�t��&@aá/EX?iD��,�?�Unknown
�,HostBiasAddGrad"6gradient_tape/functional_1/dense_3/BiasAdd/BiasAddGrad(1�G�z�"@9�G�z�"@A�G�z�"@I�G�z�"@aV�C��S?i�� "$�?�Unknown
�-HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1NbX94@9NbX94@ANbX94@INbX94@a�A�7�P?i��gs,�?�Unknown
�.HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1�C�l��@9�C�l��@A�C�l��@I�C�l��@a�&���O?i|���p4�?�Unknown
�/HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1����S@9����S@A����S@I����S@ay�f#3N?i+I���;�?�Unknown
0HostMul".gradient_tape/dense_2/kernel/Regularizer/Mul_1(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a�ɟ4
M?ip;�@C�?�Unknown
u1HostSquare"!dense_2/kernel/Regularizer/Square(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a
z|XF^K?i�ZS�J�?�Unknown
�2HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a撃�u�I?it{ˍ�P�?�Unknown
�3HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1`��"�y@9`��"�y@A`��"�y@I`��"�y@a�9�I?iB�J�V�?�Unknown
s4HostDataset"Iterator::Model::ParallelMapV2(1���Sc@9���Sc@A���Sc@I���Sc@a�j3���H?i��]�?�Unknown
X5HostAddN"AddN_5(1��n��@9��n��@A��n��@I��n��@a����\AH?i�K�]c�?�Unknown
g6HostStridedSlice"strided_slice(1�I+�@9�I+�@A�I+�@I�I+�@axp�P�F?i,O%2�h�?�Unknown
}7HostMul",gradient_tape/dense/kernel/Regularizer/Mul_1(1R����@9R����@AR����@IR����@ay�$N2F?ij���dn�?�Unknown
�8HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1ףp=
W@9ףp=
W@Aףp=
W@Iףp=
W@a=v��E?i�5�}�s�?�Unknown
r9HostTensorSliceDataset"TensorSliceDataset(1ףp=
�@9ףp=
�@Aףp=
�@Iףp=
�@a�a�&E?i1�q&y�?�Unknown
�:HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1h��|?5@9h��|?5@A�&1�@I�&1�@a&�z��_D?iܹa#2~�?�Unknown
X;HostAddN"AddN_4(1������@9������@A������@I������@a�Ty��D?i1X�5��?�Unknown
o<HostSum"dense_2/kernel/Regularizer/Sum(1T㥛Ġ@9T㥛Ġ@AT㥛Ġ@IT㥛Ġ@a{	>��A?i�������?�Unknown
�=HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1y�&1,@9y�&1,@Ay�&1,@Iy�&1,@a����=A?itZ���?�Unknown
l>HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@a�B6|y�@?ih;b��?�Unknown
u?HostSquare"!dense_1/kernel/Regularizer/Square(1�G�z@9�G�z@A�G�z@I�G�z@a�c���@?i��?��?�Unknown
�@HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1\���(\@9\���(\@A\���(\@I\���(\@a4+��.@?iiW_K��?�Unknown
[AHostAddV2"Adam/add(1��K7�@9��K7�@A��K7�@I��K7�@a�0��|??im]7�:��?�Unknown
{BHostSum"*categorical_crossentropy/weighted_loss/Sum(1ףp=
�
@9ףp=
�
@Aףp=
�
@Iףp=
�
@aX��
�<?i�Y{�Ο�?�Unknown
VCHostSum"Sum_2(1��� �r
@9��� �r
@A��� �r
@I��� �r
@ac��2<?i�M�T��?�Unknown
XDHostEqual"Equal(1^�I+
@9^�I+
@A^�I+
@I^�I+
@ajgS��;?i�E͐Ѧ�?�Unknown
eEHost
LogicalAnd"
LogicalAnd(1m�����	@9m�����	@Am�����	@Im�����	@a�%�;?i���B��?�Unknown�
YFHostPow"Adam/Pow(1333333	@9333333	@A333333	@I333333	@a_�yu�:?i'�7���?�Unknown
�GHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1��x�&1@9��x�&1�?A��x�&1@I��x�&1�?az�L\�9?i��F�װ�?�Unknown
XHHostSlice"Slice(1)\���(@9)\���(@A)\���(@I)\���(@a2�����9?i�X���?�Unknown
xIHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�n��`@9�n��`@A�x�&1@I�x�&1@a|���9?i�I��C��?�Unknown
rJHostConcatenateDataset"ConcatenateDataset(1� �rh�@9� �rh�@A� �rh�@I� �rh�@a��� 9?i7���g��?�Unknown
vKHostAssignAddVariableOp"AssignAddVariableOp_2(1'1�Z@9'1�Z@A'1�Z@I'1�Z@a����]�5?i�ZD���?�Unknown
�LHostReadVariableOp"*functional_1/dense_2/MatMul/ReadVariableOp(1D�l���@9D�l���@AD�l���@ID�l���@a�x���4?ieց����?�Unknown
XMHostCast"Cast_1(1�t�V@9�t�V@A�t�V@I�t�V@a .���4?i�+;QP��?�Unknown
tNHostReadVariableOp"Adam/Cast/ReadVariableOp(1��ʡE�@9��ʡE�@A��ʡE�@I��ʡE�@a���@��3?i>c����?�Unknown
mOHostSum"dense/kernel/Regularizer/Sum(1㥛� �@9㥛� �@A㥛� �@I㥛� �@a�Wv<�3?i	�1L��?�Unknown
�PHostReadVariableOp"(functional_1/dense/MatMul/ReadVariableOp(1��Q�@9��Q�@A��Q�@I��Q�@af��n8Q3?i���X���?�Unknown
oQHostSum"dense_1/kernel/Regularizer/Sum(1���Q�@9���Q�@A���Q�@I���Q�@aެ���2?i_�����?�Unknown
[RHostPow"
Adam/Pow_1(1u�V@9u�V@Au�V@Iu�V@a�k��.2?iÌ��X��?�Unknown
�SHostReadVariableOp"*functional_1/dense_3/MatMul/ReadVariableOp(1��� �r @9��� �r @A��� �r @I��� �r @a%m|�1?iQ��҉��?�Unknown
�THostReadVariableOp"+functional_1/dense_2/BiasAdd/ReadVariableOp(1sh��|? @9sh��|? @Ash��|? @Ish��|? @a�ۈ�RR1?il�����?�Unknown
ZUHostSlice"Slice_1(1J+� @9J+� @AJ+� @IJ+� @aw4�6�&1?i�������?�Unknown
�VHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1���Q��?9���Q��?A���Q��?I���Q��?a��ޕ�_0?ilH+����?�Unknown
wWHostReadVariableOp"div_no_nan_1/ReadVariableOp(1B`��"��?9B`��"��?AB`��"��?IB`��"��?avmO�&�/?ic��3���?�Unknown
�XHostReadVariableOp".dense/kernel/Regularizer/Square/ReadVariableOp(1��MbX�?9��MbX�?A��MbX�?I��MbX�?a�P6kH/?ilbJ����?�Unknown
tYHostAssignAddVariableOp"AssignAddVariableOp(17�A`���?97�A`���?A7�A`���?I7�A`���?aH�)�Q�.?i�g?���?�Unknown
bZHostDivNoNan"div_no_nan_1(1X9��v�?9X9��v�?AX9��v�?IX9��v�?a��6jl6,?isX.����?�Unknown
s[HostSquare"dense/kernel/Regularizer/Square(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a0��[S+?i�y��:��?�Unknown
v\HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1��Q���?9��Q���?A��Q���?I��Q���?a2� ��4+?i	J�(���?�Unknown
v]HostAssignAddVariableOp"AssignAddVariableOp_4(1j�t��?9j�t��?Aj�t��?Ij�t��?a��#)�)?i�{&+���?�Unknown
h^HostTensorDataset"TensorDataset(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?a>�\\G)?i�q���?�Unknown
X_HostAddN"AddN_3(1�l�����?9�l�����?A�l�����?I�l�����?aw`�7�u(?ix�!����?�Unknown
o`HostReadVariableOp"Adam/ReadVariableOp(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a�{߲�'?ip��#��?�Unknown
~aHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1��|?5^�?9��|?5^�?A��|?5^�?I��|?5^�?aa�?B��'?ijA�&���?�Unknown
obHostMul"dense_2/kernel/Regularizer/mul(1����K�?9����K�?A����K�?I����K�?a��ɣ��&?i	~�f��?�Unknown
]cHostCast"Adam/Cast_1(1�(\����?9�(\����?A�(\����?I�(\����?a����LX&?i4�d�q��?�Unknown
VdHostCast"Cast(1NbX9��?9NbX9��?ANbX9��?INbX9��?a�R��#?i-�	����?�Unknown
zeHostAddN"(ArithmeticOptimizer/AddOpsRewrite_AddN_1(1��/�$�?9��/�$�?A��/�$�?I��/�$�?a�9�W#?i�~]q���?�Unknown
XfHostCast"Cast_2(1�t�V�?9�t�V�?A�t�V�?I�t�V�?a'�dWj!?i5����?�Unknown
�gHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1��C�l��?9��C�l��?A��C�l��?I��C�l��?a��g�!?i\�t/��?�Unknown
uhHostReadVariableOp"div_no_nan/ReadVariableOp(1�I+��?9�I+��?A�I+��?I�I+��?a���l�E ?iL~ۋ��?�Unknown
�iHostReadVariableOp"+functional_1/dense_3/BiasAdd/ReadVariableOp(1��"��~�?9��"��~�?A��"��~�?I��"��~�?a�ǐhA ?i��d���?�Unknown
�jHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1�ʡE���?9�ʡE���?A�ʡE���?I�ʡE���?aO8?�Y�?i��3��?�Unknown
�kHostDivNoNan",categorical_crossentropy/weighted_loss/value(1�x�&1�?9�x�&1�?A�x�&1�?I�x�&1�?a]_��Z�?i�8	(��?�Unknown
TlHostMul"Mul(1V-���?9V-���?AV-���?IV-���?a�EtU�p?is䛮���?�Unknown
�mHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1T㥛Ġ@9T㥛Ġ@A/�$���?I/�$���?aiW$�h?i��P����?�Unknown
�nHostReadVariableOp")functional_1/dense/BiasAdd/ReadVariableOp(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?a�'�"~�?i��AC���?�Unknown
�oHostReadVariableOp"+functional_1/dense_1/BiasAdd/ReadVariableOp(1#��~j��?9#��~j��?A#��~j��?I#��~j��?aG�Z��^?i�/�9���?�Unknown
vpHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?ao�j`�D?i3�^T��?�Unknown
`qHostDivNoNan"
div_no_nan(1�O��n�?9�O��n�?A�O��n�?I�O��n�?a���?i�q$��?�Unknown
vrHostAssignAddVariableOp"AssignAddVariableOp_1(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a�6��?i������?�Unknown
vsHostAssignAddVariableOp"AssignAddVariableOp_3(1u�V�?9u�V�?Au�V�?Iu�V�?a�ur�r?i �6���?�Unknown
mtHostMul"dense/kernel/Regularizer/mul(1�Q����?9�Q����?A�Q����?I�Q����?a">�.�?i����?�Unknown
�uHostReadVariableOp"0dense_1/kernel/Regularizer/Square/ReadVariableOp(1��(\���?9��(\���?A��(\���?I��(\���?a��I���?i7�サ��?�Unknown
ovHostMul"dense_1/kernel/Regularizer/mul(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a��	�	�?i�A1�K��?�Unknown
�wHostReadVariableOp"0dense_2/kernel/Regularizer/Square/ReadVariableOp(1sh��|?�?9sh��|?�?Ash��|?�?Ish��|?�?a4�J=c?i������?�Unknown
�xHostReadVariableOp"*functional_1/dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a0e��?iǤ9rg��?�Unknown
�yHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1��K7��?9��K7��?A��K7��?I��K7��?aW�(�(9?ij��V���?�Unknown
wzHostReadVariableOp"div_no_nan/ReadVariableOp_1(1����Mb�?9����Mb�?A����Mb�?I����Mb�?a,C���	?iA�Q<��?�Unknown
a{HostIdentity"Identity(1��x�&1�?9��x�&1�?A��x�&1�?I��x�&1�?az�L\�	?iFqW{���?�Unknown�
y|HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1V-��?9V-��?AV-��?IV-��?at�#*!?i     �?�Unknown*�v
sHostCast"!functional_1/dropout/dropout/Cast(1�Zd;%q@9�Zd;%q@A�Zd;%q@I�Zd;%q@a4���h�?i4���h�?�Unknown
HostMatMul"+gradient_tape/functional_1/dense_1/MatMul_1(1�G�z�g@9�G�z�g@A�G�z�g@I�G�z�g@a���%�ڭ?i��6)+�?�Unknown
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1�Vi@9�Vi@A�G�z�f@I�G�z�f@a��ޑ ��?i�A��qM�?�Unknown
HostMatMul"+gradient_tape/functional_1/dense_2/MatMul_1(1}?5^�]f@9}?5^�]f@A}?5^�]f@I}?5^�]f@a�U����?iz���s$�?�Unknown
}HostMatMul")gradient_tape/functional_1/dense_2/MatMul(1�z�Gid@9�z�Gid@A�z�Gid@I�z�Gid@a����|�?iR-�T�?�Unknown
hHostRandomShuffle"RandomShuffle(1!�rh�d@9!�rh�d@A!�rh�d@I!�rh�d@a����j��?i-��~s�?�Unknown
�HostRandomUniform"9functional_1/dropout/dropout/random_uniform/RandomUniform(1�|?5^�b@9�|?5^�b@A�|?5^�b@I�|?5^�b@a���Ȕ�?iǉ��f�?�Unknown
sHost_FusedMatMul"functional_1/dense_1/Relu(1�|?5^F`@9�|?5^F`@A�|?5^F`@I�|?5^F`@aE��Q�R�?i�H,fl��?�Unknown
}	HostMatMul")gradient_tape/functional_1/dense_1/MatMul(1�����,Z@9�����,Z@A�����,Z@I�����,Z@a��dc�W�?i�ޘrc��?�Unknown

HostMatMul"+gradient_tape/functional_1/dense_3/MatMul_1(1V-���X@9V-���X@AV-���X@IV-���X@a�$�we�?i� ʹ��?�Unknown
�HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1��~j��X@9��~j��X@A��~j��X@I��~j��X@ajx���?iB��K���?�Unknown
HostReluGrad")gradient_tape/functional_1/dense/ReluGrad(1��Q�^X@9��Q�^X@A��Q�^X@I��Q�^X@aߐ�Gnn�?i�追^��?�Unknown
qHost_FusedMatMul"functional_1/dense/Relu(1+��.X@9+��.X@A+��.X@I+��.X@a�A�2�?iu������?�Unknown
sHost_FusedMatMul"functional_1/dense_2/Relu(1� �rh�W@9� �rh�W@A� �rh�W@I� �rh�W@a�]&���?ie${Qs��?�Unknown
�HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1R���q]@9R���q]@A��Q��W@I��Q��W@a�Qr����?i�U��?�Unknown
�HostReluGrad"+gradient_tape/functional_1/dense_2/ReluGrad(1L7�A`�V@9L7�A`�V@AL7�A`�V@IL7�A`�V@aJ�h�3��?i�����?�Unknown
sHostMul""functional_1/dropout/dropout/Mul_1(1F�����R@9F�����R@AF�����R@IF�����R@a�y���?i�W��Q�?�Unknown
}HostMatMul")gradient_tape/functional_1/dense_3/MatMul(1u�V�P@9u�V�P@Au�V�P@Iu�V�P@a�9�q�?i�Z!���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1��� ��O@9��� ��O@A��� ��O@I��� ��O@a`��t�?iɅ�,���?�Unknown
�HostReluGrad"+gradient_tape/functional_1/dense_1/ReluGrad(1)\���hJ@9)\���hJ@A)\���hJ@I)\���hJ@a���&H}�?i�q�m��?�Unknown
qHostSoftmax"functional_1/dense_3/Softmax(133333#I@933333#I@A33333#I@I33333#I@a�dR��c�?i`�z�)��?�Unknown
vHost_FusedMatMul"functional_1/dense_3/BiasAdd(1���(\�G@9���(\�G@A���(\�G@I���(\�G@a��&_�?i�Vw
��?�Unknown
{HostMatMul"'gradient_tape/functional_1/dense/MatMul(1w��/�F@9w��/�F@Aw��/�F@Iw��/�F@a�!y�?i3ݻ"���?�Unknown
`HostGatherV2"
GatherV2_1(1��Mb�C@9��Mb�C@A��Mb�C@I��Mb�C@a�)E.�w�?i��t�y��?�Unknown
�HostGreaterEqual")functional_1/dropout/dropout/GreaterEqual(1��Q��A@9��Q��A@A��Q��A@I��Q��A@a�d�l�P�?ino'�>�?�Unknown
^HostGatherV2"GatherV2(1��� ��@@9��� ��@@A��� ��@@I��� ��@@a*����?i7fj�s��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1�x�&1�@@9�x�&1�@@A�x�&1�@@I�x�&1�@@a�_`yդ�?i��O���?�Unknown
qHostMul" functional_1/dropout/dropout/Mul(1�z�G�>@9�z�G�>@A�z�G�>@I�z�G�>@a��{{�3�?i��=��1�?�Unknown
�HostMul"0gradient_tape/functional_1/dropout/dropout/Mul_1(1�Zd�=@9�Zd�=@A�Zd�=@I�Zd�=@a��	�7��?i��Eܶ|�?�Unknown
\HostArgMax"ArgMax_1(1����S=@9����S=@A����S=@I����S=@a�/�J�O�?i��q����?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(11�Z�9@91�Z�9@A1�Z�9@I1�Z�9@a��$��?ia�qʯ�?�Unknown�
Z HostArgMax"ArgMax(1�t�68@9�t�68@A�t�68@I�t�68@aa�O�;~?i6]�&B�?�Unknown
�!HostBiasAddGrad"6gradient_tape/functional_1/dense_1/BiasAdd/BiasAddGrad(1�G�z�5@9�G�z�5@A�G�z�5@I�G�z�5@a��u�mb{?i#I���x�?�Unknown
�"HostBiasAddGrad"6gradient_tape/functional_1/dense_2/BiasAdd/BiasAddGrad(1�rh���4@9�rh���4@A�rh���4@I�rh���4@aA�9�`�y?i �V����?�Unknown
#HostMul".gradient_tape/dense_1/kernel/Regularizer/Mul_1(1sh��|_4@9sh��|_4@Ash��|_4@Ish��|_4@a@^���py?i�!����?�Unknown
�$HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1�v���2@9�v���2@A�v���2@I�v���2@a�#+)�Tw?i#x�A�?�Unknown
�%HostBiasAddGrad"4gradient_tape/functional_1/dense/BiasAdd/BiasAddGrad(1;�O���2@9;�O���2@A;�O���2@I;�O���2@a��|!Tw?i's���<�?�Unknown
�&HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1+��η1@9+��η1@Aj�t��/@Ij�t��/@a����1�s?i�^62^d�?�Unknown
i'HostWriteSummary"WriteSummary(1/�$�-@9/�$�-@A/�$�-@I/�$�-@ag���kr?i����5��?�Unknown�
d(HostDataset"Iterator::Model(1�/�$�2@9�/�$�2@A��v��*@I��v��*@a��:3_Lp?iANb�Ω�?�Unknown
�)HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1�"��~�)@9�"��~�)@A��|?5)@I��|?5)@aJ~�ҋ]o?i��4,��?�Unknown
�*HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1j�t��&@9j�t��&@Aj�t��&@Ij�t��&@a����ml?ie�˙��?�Unknown
�+HostBiasAddGrad"6gradient_tape/functional_1/dense_3/BiasAdd/BiasAddGrad(1�G�z�"@9�G�z�"@A�G�z�"@I�G�z�"@a����}Sg?iF��H���?�Unknown
�,HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1NbX94@9NbX94@ANbX94@INbX94@a�
~��{c?iQ)��h�?�Unknown
�-HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1�C�l��@9�C�l��@A�C�l��@I�C�l��@aA� ���b?i�I9\!#�?�Unknown
�.HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1����S@9����S@A����S@I����S@a7{U&ٯa?ip�_5�4�?�Unknown
/HostMul".gradient_tape/dense_2/kernel/Regularizer/Mul_1(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a���5�a?iVy�'�E�?�Unknown
u0HostSquare"!dense_2/kernel/Regularizer/Square(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a����R`?i�s%z�U�?�Unknown
�1HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a��U��D^?i�����d�?�Unknown
�2HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1`��"�y@9`��"�y@A`��"�y@I`��"�y@a��>��P]?i�?<�s�?�Unknown
s3HostDataset"Iterator::Model::ParallelMapV2(1���Sc@9���Sc@A���Sc@I���Sc@a�f�4]?ir��}?��?�Unknown
X4HostAddN"AddN_5(1��n��@9��n��@A��n��@I��n��@a;�+3i\?i��[t��?�Unknown
g5HostStridedSlice"strided_slice(1�I+�@9�I+�@A�I+�@I�I+�@a�*����Z?i�#��?�Unknown
}6HostMul",gradient_tape/dense/kernel/Regularizer/Mul_1(1R����@9R����@AR����@IR����@a\ۈw��Y?i�����?�Unknown
�7HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1ףp=
W@9ףp=
W@Aףp=
W@Iףp=
W@a�u�&fY?i�C�뗷�?�Unknown
r8HostTensorSliceDataset"TensorSliceDataset(1ףp=
�@9ףp=
�@Aףp=
�@Iףp=
�@a�1�O�X?i�\����?�Unknown
�9HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1h��|?5@9h��|?5@A�&1�@I�&1�@a�"P��W?i�e����?�Unknown
X:HostAddN"AddN_4(1������@9������@A������@I������@a�܃!{W?i?�&p���?�Unknown
o;HostSum"dense_2/kernel/Regularizer/Sum(1T㥛Ġ@9T㥛Ġ@AT㥛Ġ@IT㥛Ġ@al�����T?i��3	��?�Unknown
�<HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1y�&1,@9y�&1,@Ay�&1,@Iy�&1,@a��Ji�1T?i��<"��?�Unknown
l=HostIteratorGetNext"IteratorGetNext(1333333@9333333@A333333@I333333@aWiP�zS?i2�C����?�Unknown
u>HostSquare"!dense_1/kernel/Regularizer/Square(1�G�z@9�G�z@A�G�z@I�G�z@a�|��gS?i�34b��?�Unknown
�?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1\���(\@9\���(\@A\���(\@I\���(\@aZ��Т�R?i褜��?�Unknown
[@HostAddV2"Adam/add(1��K7�@9��K7�@A��K7�@I��K7�@aS�)D�pR?i繾*F�?�Unknown
{AHostSum"*categorical_crossentropy/weighted_loss/Sum(1ףp=
�
@9ףp=
�
@Aףp=
�
@Iףp=
�
@a����P?iO�8,��?�Unknown
VBHostSum"Sum_2(1��� �r
@9��� �r
@A��� �r
@I��� �r
@a���
[�P?i����&�?�Unknown
XCHostEqual"Equal(1^�I+
@9^�I+
@A^�I+
@I^�I+
@ab���VP?i9�&/�?�Unknown
eDHost
LogicalAnd"
LogicalAnd(1m�����	@9m�����	@Am�����	@Im�����	@a5���P?i�o��#7�?�Unknown�
YEHostPow"Adam/Pow(1333333	@9333333	@A333333	@I333333	@af[�e�wO?i��a�?�?�Unknown
�FHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1��x�&1@9��x�&1�?A��x�&1@I��x�&1�?a9ll�5N?i6�<H�F�?�Unknown
XGHostSlice"Slice(1)\���(@9)\���(@A)\���(@I)\���(@a`T4�L+N?iK9gN�?�Unknown
xHHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�n��`@9�n��`@A�x�&1@I�x�&1@a��˝aN?iH�γ�U�?�Unknown
rIHostConcatenateDataset"ConcatenateDataset(1� �rh�@9� �rh�@A� �rh�@I� �rh�@a6�q�nM?i��37�\�?�Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_2(1'1�Z@9'1�Z@A'1�Z@I'1�Z@a�>��iI?i=9�Pc�?�Unknown
�KHostReadVariableOp"*functional_1/dense_2/MatMul/ReadVariableOp(1D�l���@9D�l���@AD�l���@ID�l���@a�=���H?i^g9ti�?�Unknown
XLHostCast"Cast_1(1�t�V@9�t�V@A�t�V@I�t�V@av�Fk2%H?i�/�}o�?�Unknown
tMHostReadVariableOp"Adam/Cast/ReadVariableOp(1��ʡE�@9��ʡE�@A��ʡE�@I��ʡE�@as����]G?i>'�Tu�?�Unknown
mNHostSum"dense/kernel/Regularizer/Sum(1㥛� �@9㥛� �@A㥛� �@I㥛� �@a�f��VG?i�{Gw*{�?�Unknown
�OHostReadVariableOp"(functional_1/dense/MatMul/ReadVariableOp(1��Q�@9��Q�@A��Q�@I��Q�@aH?�~y�F?i�5��Ҁ�?�Unknown
oPHostSum"dense_1/kernel/Regularizer/Sum(1���Q�@9���Q�@A���Q�@I���Q�@a2A�� F?i�f�Z��?�Unknown
[QHostPow"
Adam/Pow_1(1u�V@9u�V@Au�V@Iu�V@a`��WLE?i�
Rҭ��?�Unknown
�RHostReadVariableOp"*functional_1/dense_3/MatMul/ReadVariableOp(1��� �r @9��� �r @A��� �r @I��� �r @aI��9��D?iσ QА�?�Unknown
�SHostReadVariableOp"+functional_1/dense_2/BiasAdd/ReadVariableOp(1sh��|? @9sh��|? @Ash��|? @Ish��|? @a>��JD?i����?�Unknown
ZTHostSlice"Slice_1(1J+� @9J+� @AJ+� @IJ+� @aU�)�D?i������?�Unknown
�UHostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1���Q��?9���Q��?A���Q��?I���Q��?a}��X-.C?i�6����?�Unknown
wVHostReadVariableOp"div_no_nan_1/ReadVariableOp(1B`��"��?9B`��"��?AB`��"��?IB`��"��?a�tl�B?i��]��?�Unknown
�WHostReadVariableOp".dense/kernel/Regularizer/Square/ReadVariableOp(1��MbX�?9��MbX�?A��MbX�?I��MbX�?aɵ��=RB?iϸ����?�Unknown
tXHostAssignAddVariableOp"AssignAddVariableOp(17�A`���?97�A`���?A7�A`���?I7�A`���?aq4��A?i��#q��?�Unknown
bYHostDivNoNan"div_no_nan_1(1X9��v�?9X9��v�?AX9��v�?IX9��v�?a��E��@?iL�>����?�Unknown
sZHostSquare"dense/kernel/Regularizer/Square(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a�N2�� @?i��ْ��?�Unknown
v[HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1��Q���?9��Q���?A��Q���?I��Q���?a�Ii�??i	it����?�Unknown
v\HostAssignAddVariableOp"AssignAddVariableOp_4(1j�t��?9j�t��?Aj�t��?Ij�t��?a�$�#�>?i�XvQ��?�Unknown
h]HostTensorDataset"TensorDataset(1��ʡE��?9��ʡE��?A��ʡE��?I��ʡE��?a�G�=?i����?�Unknown
X^HostAddN"AddN_3(1�l�����?9�l�����?A�l�����?I�l�����?a2�Լ��<?i���˙��?�Unknown
o_HostReadVariableOp"Adam/ReadVariableOp(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a�L3��<?i�(��?�Unknown
~`HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1��|?5^�?9��|?5^�?A��|?5^�?I��|?5^�?a�	r�;?iֲ�����?�Unknown
oaHostMul"dense_2/kernel/Regularizer/mul(1����K�?9����K�?A����K�?I����K�?a����:?iv������?�Unknown
]bHostCast"Adam/Cast_1(1�(\����?9�(\����?A�(\����?I�(\����?a��X,:?i�qz0��?�Unknown
VcHostCast"Cast(1NbX9��?9NbX9��?ANbX9��?INbX9��?a}r��)[7?iD�B���?�Unknown
zdHostAddN"(ArithmeticOptimizer/AddOpsRewrite_AddN_1(1��/�$�?9��/�$�?A��/�$�?I��/�$�?a*�
�%�6?i�������?�Unknown
XeHostCast"Cast_2(1�t�V�?9�t�V�?A�t�V�?I�t�V�?aэ�-f4?i�}��?�Unknown
�fHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1��C�l��?9��C�l��?A��C�l��?I��C�l��?a�o�ml�3?i��4���?�Unknown
ugHostReadVariableOp"div_no_nan/ReadVariableOp(1�I+��?9�I+��?A�I+��?I�I+��?a�n^}3?il��]��?�Unknown
�hHostReadVariableOp"+functional_1/dense_3/BiasAdd/ReadVariableOp(1��"��~�?9��"��~�?A��"��~�?I��"��~�?ac�_
3?i���R���?�Unknown
�iHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1�ʡE���?9�ʡE���?A�ʡE���?I�ʡE���?aT��6l�2?ieS���?�Unknown
�jHostDivNoNan",categorical_crossentropy/weighted_loss/value(1�x�&1�?9�x�&1�?A�x�&1�?I�x�&1�?a��$a��1?i���D��?�Unknown
TkHostMul"Mul(1V-���?9V-���?AV-���?IV-���?a�1�n>1?i]�l��?�Unknown
�lHostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1T㥛Ġ@9T㥛Ġ@A/�$���?I/�$���?a�%.�81?i������?�Unknown
�mHostReadVariableOp")functional_1/dense/BiasAdd/ReadVariableOp(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?a�x�*�/?i������?�Unknown
�nHostReadVariableOp"+functional_1/dense_1/BiasAdd/ReadVariableOp(1#��~j��?9#��~j��?A#��~j��?I#��~j��?a��\n�.?i8x�Ps��?�Unknown
voHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�p=
ף�?9�p=
ף�?A�p=
ף�?I�p=
ף�?a����.?i����_��?�Unknown
`pHostDivNoNan"
div_no_nan(1�O��n�?9�O��n�?A�O��n�?I�O��n�?a�)=�~�,?i�;��,��?�Unknown
vqHostAssignAddVariableOp"AssignAddVariableOp_1(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?az��*+?i7�M����?�Unknown
vrHostAssignAddVariableOp"AssignAddVariableOp_3(1u�V�?9u�V�?Au�V�?Iu�V�?a;K��K*?iL�w���?�Unknown
msHostMul"dense/kernel/Regularizer/mul(1�Q����?9�Q����?A�Q����?I�Q����?a>*=�`&?i��~����?�Unknown
�tHostReadVariableOp"0dense_1/kernel/Regularizer/Square/ReadVariableOp(1��(\���?9��(\���?A��(\���?I��(\���?a��nd-&?i���VJ��?�Unknown
ouHostMul"dense_1/kernel/Regularizer/mul(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a�Sicy&?i?�[����?�Unknown
�vHostReadVariableOp"0dense_2/kernel/Regularizer/Square/ReadVariableOp(1sh��|?�?9sh��|?�?Ash��|?�?Ish��|?�?au50A��%?iB�9��?�Unknown
�wHostReadVariableOp"*functional_1/dense_1/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?am�����#?i�*)�B��?�Unknown
�xHostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(1��K7��?9��K7��?A��K7��?I��K7��?a�U�c+�?i^I��A��?�Unknown
wyHostReadVariableOp"div_no_nan/ReadVariableOp_1(1����Mb�?9����Mb�?A����Mb�?I����Mb�?aN�*��r?i�2Ė5��?�Unknown
azHostIdentity"Identity(1��x�&1�?9��x�&1�?A��x�&1�?I��x�&1�?a9ll�5?i)��B'��?�Unknown�
y{HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1V-��?9V-��?AV-��?IV-��?a,(��?i     �?�Unknown