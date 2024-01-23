
pub mod codegen;
pub mod tokens;
pub mod ast;


/**
 * Notes
 * 
 * Collider Integration
 * 
 * - Scripts can be run as jobs on the threadpool
 * - Scripts only ever have a read-only copy of the arguments they take
 * - The final resulting value that a script wishes to write back to a component
 *      in collider is emitted as a DatabaseCommand
 * - DatabaseCommands are executed at the discretion of the consumer of collider,
 *      or collider itself. They can be executed
 * 
 * 
 */
struct __NOTES;