	.file	"multiplex.c"
	.comm	g_A,4,4
	.comm	g_B,4,4
	.comm	mutex,32,32
	.text
	.globl	workerfoo
	.type	workerfoo, @function
workerfoo:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movl	$mutex, %edi
	call	sem_wait
	movl	g_A(%rip), %eax
	addl	$1, %eax
	movl	%eax, g_A(%rip)
	movl	$mutex, %edi
	call	sem_post
	movl	g_A(%rip), %eax
	cltq
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	workerfoo, .-workerfoo
	.section	.rodata
.LC0:
	.string	"Value of g_A=%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$432, %rsp
	movl	%edi, -420(%rbp)
	movq	%rsi, -432(%rbp)
	movl	$0, g_A(%rip)
	movl	$0, g_B(%rip)
	movl	$0, %edi
	call	time
	movl	%eax, %edi
	call	srand
	movl	$10, %edx
	movl	$0, %esi
	movl	$mutex, %edi
	call	sem_init
	movl	$0, -4(%rbp)
	jmp	.L3
.L4:
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	-416(%rbp), %rax
	addq	%rdx, %rax
	movl	$0, %ecx
	movl	$workerfoo, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create
	addl	$1, -4(%rbp)
.L3:
	cmpl	$49, -4(%rbp)
	jle	.L4
	movl	$49, -4(%rbp)
	jmp	.L5
.L6:
	movl	-4(%rbp), %eax
	cltq
	movq	-416(%rbp,%rax,8), %rax
	leaq	-8(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	pthread_join
	subl	$1, -4(%rbp)
.L5:
	cmpl	$0, -4(%rbp)
	jns	.L6
	movl	g_A(%rip), %edx
	movl	$.LC0, %eax
	movl	%edx, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.ident	"GCC: (Ubuntu/Linaro 4.6.3-1ubuntu5) 4.6.3"
	.section	.note.GNU-stack,"",@progbits
